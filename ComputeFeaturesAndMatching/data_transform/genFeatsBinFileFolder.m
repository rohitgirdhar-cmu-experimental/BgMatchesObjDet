function genFeatsBinFileFolder() 
featsdir = '../../tempdata/selsearch_feats/';
outfpath = '../../tempdata/selsearch_feats_all_normalized/';
try
  matlabpool open 8;
catch
end
parfor i = 1 : 237
  feats = readFeats(fullfile(featsdir, [num2str(i) '.dat']));
  feats = normr(feats);
  feats = sparse(feats);
  saveVars(fullfile(outfpath, [num2str(i) '.mat']), feats);
  fprintf('Done for %d\n', i);
end

function feats = readFeats(fpath)
if ~exist(fpath, 'file')
  [dname, fname, fext] = fileparts(fpath);
  unix(['cd ' dname '; tar xzf ' fname fext '.tar.gz']);
end
feats = dlmread(fpath);
unix(['rm ' fpath]);

function saveVars(fpath, feats)
save(fpath, 'feats');

