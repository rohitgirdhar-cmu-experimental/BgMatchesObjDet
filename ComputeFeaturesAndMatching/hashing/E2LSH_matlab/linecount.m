function count = linecount(fpath)
[~, cmdout] = unix(['wc -l ' fpath]);
elts = strsplit(cmdout);
count = str2num(elts{1});

