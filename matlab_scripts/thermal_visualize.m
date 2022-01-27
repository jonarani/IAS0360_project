clf
clc

%folder = "../dataset/thermal_raw_20210507_full/"; % Directory of the input file
%file = "20210507_1605_3078.txt";	% Input file with JSON format
folder = "../dataset/thermal_raw_20210507_full/";
file = "20210507_1605_C088.txt";

fileName = append(folder,file);

%% Configure heatmap
frame = jsondecode('[]');


h = heatmap(frame, 'FontSize', 6);
h.Title = "Thermal array color map";
h.ColorLimits = [10 40]; % Comment out if need for automatic heatmap limits
h.Colormap = parula;    % Options: parula, cool, summer
h.GridVisible = 'on';  % Options: on, off
h.ColorbarVisible = 'on';   % Options: on, off
h.ColorMethod = 'none';

%% Process frames
i = 0;
cmin = 20;  % Manual heatmap min value
cmax = 30;  % Manual heatmap max value
fid = fopen(fileName);
while ~feof(fid)
  disp(i);
  line = fgetl(fid);
  frame = jsondecode(line);

  % Dynamic heamap min and max change. Can be disabled if needed
  cmin = min(min(frame.data));
  cmax = max(max(frame.data));

  refreshdata;
  h.ColorLimits = [cmin cmax];
  h.ColorData = frame.data;
  drawnow;
  i = i + 1;
end
fclose(fid);
