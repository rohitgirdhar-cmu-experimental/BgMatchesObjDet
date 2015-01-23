function computeDistanceFromHash()
addpath(genpath('hashing/E2LSH_matlab/'));
load('hashing/E2LSH_matlab/T_50K.mat', 'T');
outpath = '../tempdata/matches_hash_matlab/';
featCounts = dlmread('../tempdata/feat_counts.txt');
for i = 1 : 100
  fpath = fullfile('../tempdata/marked_feats/', [num2str(i) '.dat']);
  untarFile(fpath);
  matches = lshlookup(fpath, [], T); % this output is 1 indexed (from matlab)
  imgids = []; featids = [];
  for m = 1 : length(matches)
    [img, featid] = row2imid(matches(m), featCounts);
    imgids(m) = img;
    featids(m) = featid;
  end
  dlmwrite(fullfile(outpath, [num2str(i) '.txt']), imgids, '\n');
  dlmwrite(fullfile(outpath, [num2str(i) '_posn.txt']), featids, '\n');
  unix(['rm ' fpath]);
  fprintf('Done for %d\n', i);
end

function untarFile(fpath)
if exist(fpath, 'file')
  return;
end
[dpath, fname, fext] = fileparts(fpath);
cmd = ['cd ' dpath '; tar xzf ' fname fext '.tar.gz'];
unix(cmd);

function [imgid, featid] = row2imid(rownum, featCounts)
% featid returned is 0 indexed. img is 1 indexed (inline with the rest of the code)
imgid = 1;
while (rownum > featCounts(imgid))
  rownum = rownum - featCounts(imgid);
  imgid = imgid + 1;
end
featid = rownum - 1; % TO 0 index it 
