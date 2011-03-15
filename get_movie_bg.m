function bg = get_movie_bg(mname,LEN)
%Get a movie filename into a virtual 'bg' dataset
  
if ~exist('mname','var') | length(mname)==0
    mname='~/Movies/Caddyshack/Caddyshack.avi';
    mname = '~/Movies/National.Geographic.Clash.of.the.Continents.PDTV.XviD.MP3.MVGroup.org.avi';
end
if ~exist('LEN','var')
    LEN = 200;
end
offset = 100;
%offset = 0;
for i = 1:LEN
    bg{i} = @()(get_movie_frame(mname,get_movie_string(offset+i)));
    %get_movie_string(10+i)
    %bg{i} = get_movie_frame(mname,get_movie_string(100+i));
end

function str = get_movie_string(index)
hours = floor(index/(3600));
minutes = floor(index/60);
seconds = index - 3600*hours-60*minutes;
str = sprintf('%02d:%02d:%02d',hours,minutes,seconds);
