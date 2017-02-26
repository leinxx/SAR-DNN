function plot_errorbar_from_data(data_path)
  avg = zeros(1, 11);
  dev = zeros(1, 11);
  f = 'error_train,error_test,error_valid';
  f = strsplit(f,',');
  for n = 1:length(f)
    data = load([data_path '/' char(f(n)) '.txt']);
    for i = 1:11
      ic = double(i - 1) / 10;
      sub = data(data(:,4) > ic - 0.01 & data(:,4) < ic + 0.01, 3);
      if isempty(sub) == 0
          avg(i) = mean(sub);
          dev(i) = std(sub);
      end
    end
    plot_errorbar(avg, dev, '');
    export_fig tmp.pdf
    movefile('tmp.pdf', [data_path '/' char(f(n)) '.pdf'])
  end
end
