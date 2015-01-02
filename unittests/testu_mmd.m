function test_suite = testu_mmd()
%
initTestSuite;

end

function test_mmd()
    X = randn(3, 100);
    Y = randn(3, 50);
    ker = KDot();
    mm = mmd(X, Y, ker);
    assert(mm >= 0);
    %display(sprintf('mmd for independence: %.3g', mm))

end

function test_dependence()
    X = randn(2, 200);
    Y = [X(1, :); X(2, :)+randn(1, 200)];

    sig2 = meddistance(X)^2;
    ker = KGaussian(sig2);
    mm = mmd(X, Y, ker);
    assert(mm >= 0);
    %display(mm)
end



