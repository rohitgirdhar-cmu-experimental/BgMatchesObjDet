function vs = findAllBuckets(T, x)
% B = FINDALLBUCKETS(TYPE,X,I)
%
% Find, for each tablae and each point(row) in file X, its hash bucket based on i
% 
%
% The bucket numbers are returned in *rows* of B, represented as
% character array. The underlying assumption that makes this possible:
% the value of each component is an integer between -128 and 127.
%
% 
% by Rohit Girdhar, CMU-RI (2015)
% Only for e2lsh-disk


% In this case, x contains a filename and just read line by line
vs = {}; % each vs{i} is a v (= [])
nlines = linecount(x);
dim = size(T(1).I.A, 1);
nRowsAtATime = 5000;
rownum = 0;
while rownum < nlines
  st = rownum;
  en = min(rownum + nRowsAtATime - 1, nlines - 1);
  feats = dlmread(x, ' ', [st, 0, en, dim - 1]);
  feats = L2normalize(feats);
  for j = 1 : length(T)
    I = T(j).I;
    vs{j}(st + 1 : en + 1, :) = ...
      floor((double(feats) * I.A - repmat(I.b, size(feats, 1), 1)) / I.W);
  end
  rownum = en + 1;
  fprintf(2, 'Read upto %d\n', rownum);
end

% enforce the range so numbers are between 0 and 255
% note: 0/1 keys in LSH become 128/129
vs = cellfun(@(v) uint8(v+128), vs, 'UniformOutput', false);

function feats = L2normalize(feats)
n = sqrt(sum(feats .^ 2, 2));
feats = bsxfun(@rdivide, feats, n);

