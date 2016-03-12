classdef KLaplace < Kernel
    %KLAPLACE Laplace kernel
    %  exp(\| x-y \|_2 / (2^sigma2))
    
    properties (SetAccess=private)
        % sigma (width) squared
        sigma2;
    end
    
    methods
        
        function this=KLaplace(sigma2)
            assert(isnumeric(sigma2));
            if length(sigma2) == 1
                this.sigma2 = sigma2;
            else
                for i=1:length(sigma2)
                    this(i) = sigma2(i);
                end
            end
        end
        
        function Kmat = eval(this, X, Y)
            % X, Y are data matrices where each column is one instance
            assert(isnumeric(X));
            assert(isnumeric(Y));

            D = sqrt(bsxfun(@plus, sum(X.^2,1)', sum(Y.^2,1)) - 2*(X'*Y ));
            Kmat = exp(-D./(2*(this.sigma2)));
            
        end
        
        function Kvec = pairEval(this, X, Y)
            assert(isnumeric(X));
            assert(isnumeric(Y));
            
            D = sqrt(sum((X-Y).^2, 1));
            Kvec = exp(-D./(2*(this.sigma2)));

        end
        
        function s=shortSummary(this)
            s = sprintf('%s(%.3g)', mfilename, this.sigma2);
        end

        function Kmat = selfEval(this, X)
            assert(isnumeric(X));
            s2 = sum(X.^2, 1);

            D = sqrt(bsxfun(@plus, s2, s2') - 2*(X'*X ));
            Kmat = exp(-D./(2*(this.sigma2)));
        end
    end
    
    methods (Static)

    end
    
    
end

