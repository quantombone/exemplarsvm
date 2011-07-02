function M = slmetric_pw(X1, X2, mtype, varargin)
%SLMETRIC_PW Compute the metric between column vectors pairwisely
%
% [ Syntax ]
%   - M = slmetric_pw(X1, X2, mtype);
%   - M = slmetric_pw(X1, X2, mtype, ...);
%
% [ Arguments ]
%   - X1, X2:       the sample matrices
%   - mtype:        the string indicating the type of metric
%   - M:            the resulting metric matrix
%
% [ Description ]
%    - M = slmetric_pw(X1, X2, mtype) Computes the metrics between
%      column vectors of X1 and X2 pairwisely, using the metric
%      specified by mtype. 
%
%      Both X1 and X2 are matrices with each column representing a 
%      sample. X1 and X2 should have the same number of rows. Suppose
%      the size of X1 is d x n1, and the size of X2 is d x n2. Then 
%      the output metric matrix M will be of size n1 x n2, in which
%      M(i, j) is the metric value between X1(:,i) and X2(:,j).
%
%    - M = slmetric_pw(X1, X2, mtype, ...) Some metric types requires
%      extra parameters, which should be specified in params.
%
%      The supported metrics of this function are listed as follows: 
%      \{:
%        - eucdist:        Euclidean distance: 
%                          $ ||x - y|| $
%
%        - sqdist:         Square of Euclidean distance: 
%                          $ ||x - y||^2 $
%
%        - dotprod:        Canonical dot product: 
%                          $ <x,y> = x^T * y $ 
%
%        - nrmcorr:        Normalized correlation (cosine angle):
%                          $ (x^T * y ) / (||x|| * ||y||) $
%
%        - corrdist:       Normalized Correlation distance
%                          $ 1 - nrmcorr(x, y) $
%
%        - angle:          Angle between two vectors (in radian)  
%                          $ arccos (nrmcorr(x, y)) $
%        - quadfrm:        Quadratic form:  
%                          $ x^T * Q * y $
%                         Q is specified in the 1st extra parameter 
%
%        - quaddiff:       Quadratic form of difference:
%                          $ (x - y)^T * Q * (x - y) $
%                         Q is specified in the 1st extra parameter 
%
%        - cityblk:        City block distance (abssum of difference)
%                          $ sum_i |x_i - y_i| $
%
%        - maxdiff:        Maximum absolute difference  
%                          $ max_i |x_i - y_i| $
%
%        - mindiff:        Minimum absolute difference
%                          $ min_i |x_i - y_i| $
%
%        - minkowski:      Minkowski distance
%                          $ (\sum_i |x_i - y_i|^p)^(1/p) $
%                         The order p is specified in the 1st extra parameter
%
%        - wsqdist:        Weighted square of Euclidean distance     
%                          $ \sum_i w_i (x_i - y_i)^2 $
%                         the weights w is specified in 1st extra parameter 
%                         as a d x 1 column vector    
%
%        - hamming:        Hamming distance with threshold t
%                          \{
%                              ht1 = x > t
%                              ht2 = y > t
%                              d = sum(ht1 ~= ht2)                  
%                          \}
%                         use threshold t as the first extra param.
%                         (by default, t is set to zero).
%
%        - hamming_nrm:    Normalized hamming distance, which equals the
%                          ratio of the elements that differ.
%                          \{
%                              ht1 = x > t
%                              ht2 = y > t
%                              d = sum(ht1 ~= ht2) / length(ht1)                
%                          \}
%                          use threshold t as the first extra param.
%                         (by default, t is set to zero).
%
%        - intersect:      Histogram Intersection
%                           $ d = sum min(x, y) / min(sum(x), sum(y))$
%
%        - intersectdis:   Histogram intersection distance
%                           $ d = 1 - sum min(x, y) / min(sum(x), sum(y)) $
%
%        - chisq:          Chi-Square Distance
%                           $ d = sum (x(i) - y(i))^2/(2 * (x(i)+y(i))) $
%
%        - kldiv:          Kull-back Leibler divergence
%                           $ d = sum x(i) log (x(i) / y(i)) $
%
%        - jeffrey:        Jeffrey divergence
%                           $ d = KL(h1, (h1+h2)/2) + KL(h2, (h1+h2)/2) $
%      \:}
%
% [ Remarks ]
%   - Both X1 and X2 should be a matrix of numeric values, except
%     for case when metric type is 'hamming' or 'hamming_nrm'. 
%     For hamming or hamming_nrm metric, the input matrix can be logical.
%
% [ Examples ]
%   - Compute different types of metrics in pairwise manner
%     \{
%         % prepare sample matrix
%         X1 = rand(10, 100);
%         X2 = rand(10, 150);
%
%         % compute the euclidean distances (L2) 
%         % between the samples in X1 and X2
%         M = slmetric_pw(X1, X2, 'eucdist');
%
%         % compute the eucidean distances between the samples
%         % in X1 in a pairwise manner
%         M = slmetric_pw(X1, X1, 'eucdist');
%
%         % compute the city block distances (L1)
%         M = slmetric_pw(X1, X2, 'cityblk'); 
%
%         % compute the normalize correlations
%         M = slmetric_pw(X1, X2, 'nrmcorr');
%
%         % compute hamming distances
%         M = slmetric_pw(X1, X2, 'hamming', 0.5);
%         M2 = slmetric_pw((X1 > 0.5), (X2 > 0.5), 'hamming');
%         assert(isequal(M, M2));
%     \}
%
%   - Compute the parameterized metrics
%     \{
%         % compute weighted squared distances with user-supplied weights
%         weights = rand(10, 1);
%         M = slmetric_pw(X1, X2, 'wsqdist', weights);
%
%         % compute quadratic distances (x-y)^T * Q (x-y)
%         Q = rand(10, 10);
%         M = slmetric_pw(X1, X2, 'quaddiff', Q);
%
%         % compute Minkowski distance of order 3         
%         M = slmetric_pw(X1, X2, 'minkowski', 3);
%     \}
%
% [ History ]
%   - Created by Dahua Lin on Dec 06th, 2005
%   - Modified by Dahua Lin on Apr 21st, 2005
%       - regularize the error reporting
%   - Modified by Dahua Lin on Sep 11st, 2005
%       - completely rewrite the core codes based on new mex computation 
%         cores, and the runtime efficiency in both time and space is 
%         significantly increased.
%   - Modified by Dahua Lin on Jul 02, 2007
%       - rewrite the core computation based on the bsxfun introduced in
%         MATLAB R2007a
%       - rewrite the core-mex for cityblk, maxdiff, mindiff
%       - introduce new metrics: corrdist, minkowski
%   - Modified by Dahua Lin on Jul 30, 2007
%       - Add the metric types for histograms, which are originally
%         implemented in slhistmetric_pw in sltoolbox v1.
%   - Modified by Dahua Lin on Aug 16, 2007
%       - revise some of the help contents
%


%% parse and verify input arguments
error(nargchk(3, inf, nargin));
assert(ischar(mtype), 'sltoolbox:slmetric_pw:invalidarg', ...
    'The metric type should be a string.');

if strcmp(mtype, 'hamming') || strcmp(mtype, 'hamming_nrm')
    assert((isnumeric(X1) || islogical(X1)) && ndims(X1) == 2 && ...
           (isnumeric(X2) || islogical(X2)) && ndims(X2) == 2, ...
        'sltoolbox:slmetric_pw:invalidarg', 'X1 and X2 should be numeric or logical matrices.');        
else
    assert(isnumeric(X1) && ndims(X1) == 2 && isnumeric(X2) && ndims(X2) == 2, ...
        'sltoolbox:slmetric_pw:invalidarg', 'X1 and X2 should be numeric matrices.');
end

assert(isa(X2, class(X1)), ...
    'sltoolbox:slmetric_pw:invalidarg', 'X1 and X2 should be of the same class.');

if isempty(X1) || isempty(X2)
    M = [];
    return;
end


%% compute
switch mtype        
    case {'eucdist', 'sqdist'}
        checkdim(X1, X2);        
        M = bsxfun(@plus, sum(X1 .* X1, 1)', (-2) * X1' * X2);        
        M = bsxfun(@plus, sum(X2 .* X2, 1), M);        
        M(M < 0) = 0;                        
        if strcmp(mtype, 'eucdist')
            M = sqrt(M);
        end 
        
    case 'dotprod'
        checkdim(X1, X2);        
        M = X1' * X2;
                
    case {'nrmcorr', 'corrdist', 'angle'}
        checkdim(X1, X2);
        ns1 = sqrt(sum(X1 .* X1, 1));
        ns2 = sqrt(sum(X2 .* X2, 1));
        ns1(ns1 == 0) = 1;  
        ns2(ns2 == 0) = 1;
        M = bsxfun(@times, X1' * X2, 1 ./ ns1');
        M = bsxfun(@times, M, 1 ./ ns2);
        switch mtype
            case 'corrdist'
                M = 1 - M;
            case 'angle'
                M = real(acos(M));
        end
                
    case 'quadfrm'
        Q = varargin{1};       
        M = X1' * Q * X2;
        
    case 'quaddiff'
        checkdim(X1, X2);        
        Q = varargin{1};
        M = X1' * (-(Q + Q')) * X2;
        M = bsxfun(@plus, M, sum(X1 .* (Q * X1), 1)');
        M = bsxfun(@plus, M, sum(X2 .* (Q * X2), 1));        
                        
    case 'cityblk'
        checkdim(X1, X2);  
        M = pwmetrics_cimp(X1, X2, int32(1));
                        
    case 'maxdiff'
        checkdim(X1, X2); 
        M = pwmetrics_cimp(X1, X2, int32(3));
        
    case 'mindiff'
        checkdim(X1, X2);  
        M = pwmetrics_cimp(X1, X2, int32(2));
        
    case 'minkowski'
        checkdim(X1, X2);
        pord = varargin{1};
        if ~isscalar(pord)
            error('sltoolbox:slmetric_pw:invalidparam', ...
                'the mikowski order should be a scalar');
        end
        pord = cast(pord, class(X1));        
        M = pwmetrics_cimp(X1, X2, int32(4), pord);
                       
    case 'wsqdist'
        d = checkdim(X1, X2);
        w = varargin{1};
        if ~isequal(size(w), [d, 1])
            error('sltoolbox:slmetric_pw:invalidparam', ...
                'the weights should be given as a d x 1 vector.');
        end              
        wX2 = bsxfun(@times, X2, w);
        M = bsxfun(@plus, (-2) * X1' * wX2, sum(wX2 .* X2, 1));
        clear wX2;        
        wX1 = bsxfun(@times, X1, w);
        M = bsxfun(@plus, M, sum(wX1 .* X1, 1)');      
        
    case {'hamming', 'hamming_nrm'}
        checkdim(X1, X2);
        if islogical(X1) && islogical(X2)
            H1 = X1;
            H2 = X2;
        else
            if isempty(varargin)
                t = 0;
            else
                t = varargin{1};
                assert(isnumeric(t) && isscalar(t), ...
                    'sltoolbox:slmetric_pw:invalidparam', 't should be a numeric scalar.');
            end
            H1 = X1 > t;
            H2 = X2 > t;
        end
        M = pwhamming_cimp(H1, H2);
        if strcmp(mtype, 'hamming_nrm')
            M = M / size(H1, 1);
        end
        
    case 'intersect'
        checkdim(X1, X2);
        M = pwmetrics_cimp(X1, X2, int32(5));
        
    case 'intersectdis'
        checkdim(X1, X2);
        M = 1 - pwmetrics_cimp(X1, X2, int32(5));
        
    case 'chisq'
        checkdim(X1, X2);
        M = pwmetrics_cimp(X1, X2, int32(6));
        
    case 'kldiv'
        checkdim(X1, X2);
        M = pwmetrics_cimp(X1, X2, int32(7));
        
    case 'jeffrey'
        checkdim(X1, X2);
        M = pwmetrics_cimp(X1, X2, int32(8));
        
    otherwise
        error('sltoolbox:slmetric_pw:unknowntype', 'Unknown metric type %s', mtype);
        
        
end
        
%% Auxiliary function

function d = checkdim(X1, X2)

d = size(X1, 1);
if d ~= size(X2, 1)
    error('sltoolbox:slmetric_pw:sizmismatch', ...
        'X1 and X2 have different sample dimensions');
end




