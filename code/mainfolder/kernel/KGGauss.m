classdef KGGauss < Kernel
    %KGGAUSS A Gaussian kernel on mean embeddings. Equivalently an exponentiated 
    %MMD^2 kernel.
    %   - The data to be used with this kernel is a cell array 1xn where each 
    %   element is a dxn matrix (n samples of d-dimensional points), i.e., a bag.
    %   - The number of samples in each bag can be distinct.
    %
    %@author Wittawat Jitkrittum

    properties (SetAccess=private)
        % width^2 for the mean embedding. Equivalently width^2 for the Gaussian 
        % kernel used to compute the MMD.
        %
        embed_width2;

        % The width2 of the outer kernel.
        outer_width2

        % embedding kernel
        ker;
    end
    
    methods
        
        function this=KGGauss(embed_width2, outer_width2)
            assert(isnumeric(embed_width2));
            assert(isnumeric(outer_width2));
            assert(embed_width2 > 0);
            assert(outer_width2 > 0);
            this.embed_width2 = embed_width2;
            this.outer_width2 = outer_width2;
            this.ker = KGaussian(embed_width2);

        end
        
        function Kmat = eval(this, X, Y)
            % X: 1x nx cell arrays of bags
            % Y: 1x ny cell arrays of bags
            if isnumeric(X)
                X = {X};
            end
            if isnumeric(Y)
                Y = {Y};
            end
            assert(iscell(X));
            assert(iscell(Y));

            MM = KGGauss.evalMMD(X, Y, this.ker);
            Kmat = exp(-0.5*MM.^2);
        end
        
        function Kmat = selfEval(this, X)
            if isnumeric(X)
                X = {X};
            end
            assert(iscell(X));
            nx = length(X);
            if nx==1 
                Kmat = 1;
                return ;
            elseif nx==0
                Kmat = [];
                return;
            end

            MM = KGGauss.selfEvalMMD(X, this.ker);
            Kmat = exp(-0.5*MM.^2);
        end

        function Kvec = pairEval(this, X, Y)
            if isnumeric(X)
                X = {X};
            end
            if isnumeric(Y)
                Y = {Y};
            end
            assert(iscell(X));
            assert(iscell(Y));
            nx = length(X);
            ny = length(Y);
            assert(nx==ny);
            Kvec = zeros(1, nx);
            for i=1:nx
                xi = X{i};
                yi = Y{i};
                mm = mmd(xi, yi, this.ker);
                Kvec(i) = mm;
            end
            Kvec = exp(-0.5*Kvec.^2);
        end
        
        
        function s=shortSummary(this)
            s = sprintf('%s(emw2=%.3g, outw2=%.3g)', mfilename, ...
            this.embed_width2, this.outer_width2);
        end

    end %end methods
    
    methods(Static)
        function MM = evalMMD(X, Y, ker)
            % compute a matrix of the distance^2 (MMD^2)

            if isnumeric(X)
                X = {X};
            end
            if isnumeric(Y)
                Y = {Y};
            end
            assert(isa(ker, 'Kernel'));
            assert(iscell(X));
            assert(iscell(Y));

            nx = length(X);
            ny = length(Y);
            MM = zeros(nx, ny);
            % slow ... Can we improve ?
            for i=1:nx
                xi = X{i};
                for j=1:ny
                    yj = Y{j};
                    mm = mmd(xi, yj, ker);
                    MM(i, j) = mm;
                end
            end
        end

        function MM = selfEvalMMD(X, ker)
            if isnumeric(X)
                X = {X};
            end
            assert(isa(ker, 'Kernel'));
            assert(iscell(X));
            nx = length(X);
            if nx==1 
                MM = 0;
                return ;
            elseif nx==0
                MM = [];
                return;
            end

            % 0's on the diagonal
            MM = zeros(nx);
            for i=1:nx 
                xi = X{i};
                for j= (i+1):nx 
                    xj = X{j};
                    MM(i, j) = mmd(xi, xj, ker);
                end
            end
            MM = MM + MM';
        end
    end %end static methods
    
end

