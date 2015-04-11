function compute_boxes(imgsDir, imgsList, outdir)
% Compute the selective search boxes for each image in the dpath and store into a txt

addpath(genpath('SelectiveSearchCodeIJCV'));

fid = fopen(imgsList);
lst = textscan(fid, '%s\n');
lst = lst{1};
fclose(fid);

if ~exist(outdir, 'dir')
  mkdir(outdir);
end

try
    matlabpool open 16;
catch
end
parfor i = 1 : numel(lst)
    outpath = fullfile(outdir, [num2str(i) '.txt']);
    lockpath = [outpath '.lock'];

    if exist(lockpath, 'dir') || exist(outpath, 'file')
      continue;
    end
    unix(['mkdir -p ' lockpath]);

    impath_str = lst{i};
    I = imread(fullfile(imgsDir, impath_str));
    boxes = selective_search_boxes(I, true, 480);

    % remove too small boxes
    boxes2 = zeros(0, 4);
    for j = 1 : size(boxes, 1)
      if (boxes(j, 3) - boxes(j, 1)) < 25 || ...
          (boxes(j, 4) - boxes(j, 2)) < 25
          continue;
      end
      boxes2(end+1, :) = boxes(j, :);
    end

    saveBoxes(boxes2, outpath);

    unix(['rmdir ' lockpath]);
    i
end

function saveBoxes(boxes, fpath)
dlmwrite(fpath, boxes);

