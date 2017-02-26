%interpolate image analysis, scale by 10

basedir = '/home/lein/sar_dnn/dataset/gsl2014_hhv_ima';
outdir = [basedir '/ima_nemo_grid/'];
hhvdir = [basedir '/hhv/'];
imadir = [basedir '/GL_nemo_ima/'];
maskdir = [basedir '/mask_land_free/'];

mkdir(outdir);
list = dir([imadir '/*.txt']);
for i = 1 : length(list)
   date = list(i).name(1:15);
   hh = imread([hhvdir date '-HH-8by8-mat.tif']);
   mask = imread([maskdir date '-mask.tif']);
   ima = load([imadir list(i).name]);
   ima_region = zeros(size(hh),'uint8');
   r = 10;
   for j = 1: size(ima, 1)
    left = max(min(ima(j,1) - r, size(hh,2)), 1);
    right = max(min(ima(j,1) + r, size(hh, 2)), 1);
    top = max(min(ima(j,2) - r, size(hh,1)), 1);
    bottom = max(min(ima(j,2) + r, size(hh, 1)), 1);
    ima_region(int32(top):int32(bottom), int32(left): int32(right)) = 1;
   end
   %se = strel('disk', 20);
   %imerode(ima_region, se);
   [xq, yq] = meshgrid(1:size(hh,2), 1: size(hh,1));
   v = griddata(ima(:,1), ima(:,2), ima(:,3), xq, yq, 'linear');
   v = v * 100;
   v(mask ~= 0 | ima_region == 0) = 11;
   v = uint8(v);
   imwrite(v,[outdir date '.tif'])
end