function compute_boxes(imgsDir, imgsList, outdir)
% Compute the selective search boxes for each image in the dpath and store into a txt

ST = 1;
random_select_n = -1; % set this to -1 if you want all the boxes
                       % else this will randomly select as many
                       % boxes
if random_select_n ~= -1
  fprintf('WARNING:: RANDOMLY SAMPLING %d BOXES!!!!!\n', random_select_n)
  fprintf('WARNING:: RANDOMLY SAMPLING %d BOXES!!!!!\n', random_select_n)
  fprintf('WARNING:: RANDOMLY SAMPLING %d BOXES!!!!!\n', random_select_n)
end

addpath(genpath('SelectiveSearchCodeIJCV'));

fid = fopen(imgsList);
lst = textscan(fid, '%s\n');
lst = lst{1};
fclose(fid);

if ~exist(outdir, 'dir')
  mkdir(outdir);
end

try
    matlabpool open 12;
catch
end
parfor i = ST : numel(lst)
    outpath = fullfile(outdir, [num2str(i) '.txt']);
    lockpath = [outpath '.lock'];

    if exist(lockpath, 'dir') || exist(outpath, 'file')
      continue;
    end
    unix(['mkdir -p ' lockpath]);

    impath_str = lst{i};
    I = imread(fullfile(imgsDir, impath_str));
    boxes = selective_search_boxes(I, true, 256);

    % remove too small boxes
    boxes2 = zeros(0, 4);
    for j = 1 : size(boxes, 1)
      if (boxes(j, 3) - boxes(j, 1)) < 40 || ...
          (boxes(j, 4) - boxes(j, 2)) < 40
          continue;
      end
      boxes2(end+1, :) = boxes(j, :);
    end

    if random_select_n ~= -1
      sel_n = min(random_select_n, size(boxes2, 1));
      sel = randperm(size(boxes2, 1), sel_n);
      boxes2 = boxes2(sel, :);
    end

    saveBoxes(boxes2, outpath);

    unix(['rmdir ' lockpath]);
    i
end

function saveBoxes(boxes, fpath)
dlmwrite(fpath, boxes);

