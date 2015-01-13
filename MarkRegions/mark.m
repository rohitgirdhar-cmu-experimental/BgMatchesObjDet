function mark(imgsdir, imgslist, outdir)
fid = fopen(imgslist);
lst = textscan(fid, '%s\n');
lst = lst{1};
fclose(fid);

fig = figure;
for i = 1 : numel(lst)
    outf = fullfile(outdir, [num2str(i) '.txt']);
    if exist(outf, 'file')
        fprintf(2, 'Already done for %d\n', i);
        continue;
    end

    I = imread(fullfile(imgsdir, lst{i}));
    imshow(I);
    rect = getrect(fig);
    fid = fopen(outf, 'w');
    % write in Selective Search format of y1,x1,y2,x2
    fprintf(fid, '%f,%f,%f,%f\n', rect(2), rect(1), rect(4) + rect(2), rect(3) + rect(1));
    fclose(fid);
end

