clc;
clear;

%% write predicted label to sphere
raw_surface = mvtk_read('/media/fenqiang/DATA/unc/Data/SmoothedThickness/Year1/T0085-1-2-1yr_lh.AlignedToUNCAtlasResampled.SphereSurf.vtk');

path =  strcat('/home/fenqiang/Spherical_U-Net/pred/pred_0.txt');
pred = load(path);

pred = struct('vertices',raw_surface.vertices(1:40962,:), 'faces',faces_40962, 'sulc', raw_surface.sulc(1:40962),'thickness',raw_surface.thickness(1:40962), 'pred_label', pred);
mvtk_write(pred, strcat('/media/fenqiang/DATA/unc/Data/predicted/', 'T0085-1-2.vtk'), 'legacy-ascii')

