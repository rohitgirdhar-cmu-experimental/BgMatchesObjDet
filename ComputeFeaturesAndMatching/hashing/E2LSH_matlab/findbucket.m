function v =findbucket(type,x,I)
% B = FINDBUCKET(TYPE,X,I)
%
% Find, for each point(column) in X, its hash bucket based on i
% 
%
% The bucket numbers are returned in *rows* of B, represented as
% character array. The underlying assumption that makes this possible:
% the value of each component is an integer between -128 and 127.
%
% 
% (C) Greg Shakhnarovich, TTI-Chicago  (2008)
% Modified by Rohit Girdhar, CMU-RI (2015)


switch type,
 case 'lsh',
  v = x(I.d,:)' <= repmat(I.t,size(x,2),1);
  
 case 'e2lsh',
  v = floor((double(x)'*I.A - repmat(I.b,size(x,2),1))/I.W);
  
 case 'e2lsh-disk',
  % In this case, x contains a filename and just read line by line
  v = [];
  nlines = linecount(x);
  dim = size(I.A, 1);
  nRowsAtATime = 5000;
  rownum = 0;
  while rownum < nlines
    st = rownum;
    en = min(rownum + nRowsAtATime - 1, nlines - 1);
    feats = dlmread(x, ' ', [st, 0, en, dim - 1]);
    feats = L2normalize(feats);
    v(st + 1 : en + 1, :) = ...
      floor((double(feats) * I.A - repmat(I.b, size(feats, 1), 1)) / I.W);
    rownum = en + 1;
    fprintf(2, 'Read upto %d\n', rownum);
  end
end

% enforce the range so numbers are between 0 and 255
% note: 0/1 keys in LSH become 128/129
v = uint8(v+128);

function feats = L2normalize(feats)
n = sqrt(sum(feats .^ 2, 2));
feats = bsxfun(@rdivide, feats, n);

