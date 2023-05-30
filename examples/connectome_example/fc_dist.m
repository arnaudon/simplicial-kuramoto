function d = fc_dist(A,B)
    d = corr(A(triu(true(size(A)),1)),B(triu(true(size(B)),1)));
end