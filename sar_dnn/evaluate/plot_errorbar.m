function plot_errorbar(m,s,which_set)
    %plot the error bar figure
    %figure('visible','off');
    figure;
    hold on
    barwidth=0.51;
    x = 0:0.1:1;
    %plot up bound 
    bar(x,s+m,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none','BarWidth',barwidth)
    
    %plot average bar
    thick=0.005;
    bar(x,m+thick,'FaceColor','r','EdgeColor','none','BarWidth',barwidth-0.01) % 0.01 narrower to make sure no red board appears in bar plot
    bar(x,m-thick,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none','BarWidth',barwidth)

    %olot lower bound,
    bar(x,m-s,'FaceColor','w','EdgeColor','none','BarWidth',barwidth+0.05)
    %plot lower bound when it is <0
    index = find(m-s>=0);
    m(index)=0;
    s(index)=0;
    h = bar(x,m-s,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none','BarWidth',barwidth);
    baseline_handle = get(h,'BaseLine');
    set(baseline_handle,'Color','w');
    %plot(0.05:0.1:1.05,0,'Color','w','LineWidth',1)
    plot(0:0.1:1,0:0.1:1,'--b','LineWidth',1)
    set(gcf,'Color','w')
    
    
    axis equal tight
    ylim([-0.1,1])
    xtick = 0:0.2:1;
    set(gca,'XTick',xtick);
    set(gca,'FontName','Helvetica');
    set(gca,'FontSize',22);
    xlabel('Image analysis', 'FontSize',22);
    ylabel('Estimation from CNN', 'FontSize',22);
    set(gca,...
    'Box'         ,'off',...
    'TickDir'     ,'out',...
    'TickLength'  , [.02 .02] , ...
    'XColor'      , [.3 .3 .3], ...
    'YColor'      , [.3 .3 .3], ...
    'YTick'       , 0:0.2:1, ...
    'XTick'       , 0:0.2:1, ...
    'LineWidth'   , 1         );
    %export_fig tmp.pdf
    %copyfile('tmp.pdf',['~/Dropbox/WorkSVN/2015-03-CNN-SFCRF-ice-RSE/figures/','error_',which_set,'.pdf'])
end

function level_mean_std(data_path)
  avg = zeros(1, 11);
  dev = zeros(1, 11);
  data = load(data_path);
  for i = 0:0.1:1
    sub = data(data(:,2) > i - 0.01 & data(:,2) < i + 0.01, 1);
    avg(i * 10 + 1) = mean(sub);
    dev(i * 10 + 1) = std(sub);
  end
end
