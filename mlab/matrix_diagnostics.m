function metrics = matrix_diagnostics(A)
% Singular values measure conditioning; signed eigenvalues measure curvature.
assert(ismatrix(A) && size(A,1) == size(A,2) && all(isfinite(A), 'all'), ...
    'Diagnostics require a finite square matrix.');
A = double(A);
s = svd(A);
tol = max(size(A)) * eps * s(1);
r = sum(s > tol);
ev = eig((A + A.') / 2);
kappa = Inf;
if r == size(A,1), kappa = s(1)/s(end); end
metrics = struct('condition', kappa, 'sigma_min', s(end), 'sigma_max', s(1), ...
    'rank', r, 'rank_tolerance', tol, 'singular', double(r < size(A,1)), ...
    'eigen_min', min(ev), 'eigen_max', max(ev), 'negative_count', sum(ev < -tol), ...
    'symmetry_error', norm(A-A.','fro') / max(norm(A,'fro'), realmin));
end
