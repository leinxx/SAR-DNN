dir=${1}
matlab=/home/lein/MATLAB/MATLAB_Production_Server/R2015a/bin/matlab
$matlab -nodisplay -r "plot_errorbar_from_data('$dir'); exit"

