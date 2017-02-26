%visualize the image analysis
function viewimagsl(date)
%close all
prefix = '/media/diskb/sar_dnn/dataset/gsl2014_hhv_ima/hhv/';
subfix = '-HH-8by8-mat.tif';
%subfix = '.tif';
%subfix = '-cls-adjusted.tif';
hhfile = [prefix,num2str(date),subfix];
im = imread(hhfile);
%im =imread(['mask/' date '-mask.tif']);
[h,w] = size(im);

imadir='/media/diskb/sar_dnn/dataset/gsl2014_hhv_ima/ima/';
imafiles = dir([imadir date '*']);
data = [];
%for i = 1: length(imafiles)
%    data = [data; load([imadir imafiles(i).name])];
%end
data = load([imadir '/' date '_ima.txt']);
%data = load(['~/sar_dnn/dataset/gsl2014_hhv_ima/batches_land_free_45/predict_l2/' date '.predict.txt']);
%s = int32(data(:,3) * 10 + 0.5);
%data = [data(:,1) data(:,2) double(s)/10.0 - data(:,4)]
%figure
%hist(data(:,3),0:0.1:1)
numel(data(:,1))
figure;

cmal = colormap('Jet');%get colormap , and maually map scatter plots to different colors
im = repmat(im,[1,1,3]);
ribon = flipud(cmal);
ribon_w = 40;
ribon = uint8(zeros(size(im,1),ribon_w,3));
for i = 1:size(ribon,1)
    index = uint8((size(ribon,1)-i)/size(ribon,1)*size(cmal,1));
    if index == 0
        index = 1;
    end
    ribon(i,:,:) = 255*reshape(repmat(cmal(index,:),[ribon_w,1]),1,ribon_w,3);
end

showribon = false;
if showribon
whitespace = uint8(zeros(size(im,1),ribon_w,3))+255;
im = [im whitespace ribon];
imshow(ribon)
end

imshow(im)
%figure;
%im=imread(['~/Work/Sea_ice/images_gsl2014/' date '_8by8.tif']);
%imshow(im)
hold on
for i = 0:0.01:1
    index = data(:,3) > i-0.005 & data(:,3) <= i+0.005 ;
    color = cmal(int32(i*(size(cmal,1)-1)+1),:);
    %color = cmal(int32((i+1)/2*(size(cmal,1)-1)+1),:);
    scatter(data(index,1),data(index,2),20,color,'filled');
end
%title(date)
set(gcf,'Color','w')
axis off

if showribon
for i = 0:0.1:1
    text(size(im,2)+5,(1-i)*size(im,1),num2str(i),'fontsize',24);
end

end
%export_fig tmp.png

%if showribon
%    movefile('tmp.png',[date '_nemo_ima.png'])
%else 
%   movefile('tmp.png',['GL_nemo_ima_view/' date '_ima.png'])
%title(num2str(date))
%ribon = repmat(0:255,[20,1])'/255.0;
%set(gca,'ydir','reverse');
%legend('1','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0');
end
