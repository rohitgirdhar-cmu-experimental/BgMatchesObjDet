function compute_boxes(imgsDir, imgsList)
% Compute the selective search boxes for each image in the dpath and store into a mat

addpath(genpath('SelectiveSearchCodeIJCV'));

fid = fopen(imgsList);
lst = textscan(fid, '%s\n');
lst = lst{1};
fclose(fid);

try
    matlabpool open 6;
catch
end
parfor i = 1 : numel(lst)
    impath_str = lst{i};
    I = imread(fullfile(imgsDir, impath_str));
    boxes{i} = selective_search_boxes(I, true, 512);
    i
end

disp('Saving to disk');
mkdir('../tempdata');
%save('../tempdata/selsearch_boxes.mat', 'boxes');
saveTxt('../tempdata/selsearch_boxes/', boxes);

function saveTxt(dpath, boxes)
mkdir(dpath);
for i = 1 : numel(boxes)
    dlmwrite(fullfile(dpath, [num2str(i) '.txt']), boxes{i});
end

