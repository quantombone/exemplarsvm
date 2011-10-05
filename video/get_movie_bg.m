function bg = get_movie_bg(mname,LEN,func)
%% Create a virtual movie dataset of frames by splitting up the movie
%% pointed to by filename mname into LEN frames
%% The LEN frames are obtained by equal-time spaced partitions
%% in a "linspace" style
%% bg contains a cell array of function calls, and images can by
%% extracted by calling I = convert_to_I(bg{i})
%% Tomasz Malisiewicz (tomasz@cmu.edu)
  
if ~exist('mname','var') | length(mname)==0
    mname='~/Movies/Caddyshack/Caddyshack.avi';
    mname = '~/Movies/National.Geographic.Clash.of.the.Continents.PDTV.XviD.MP3.MVGroup.org.avi';
end

%FFMPEG_PATH = '/nfs/baikal/tmalisie/ffmpeg/ffmpeg-0.6/ffmpeg';
FFMPEG_PATH = 'ffmpeg';

%First we determine the length of the movie
instring = sprintf([FFMPEG_PATH ' -i' ...
                    ' "%s" 2>&1 | grep Duration'], mname);
[tmp,lenstring] = unix(instring);


commas = find(lenstring==',');
commas = commas(1);
f = strfind(lenstring,'Duration:');
lenstring = lenstring(f+10:commas-1);
seps = find(lenstring==':');
hour = sscanf(lenstring(1:seps(1)-1),'%d');
minute = sscanf(lenstring(seps(1)+1:seps(2)-1),'%d');
second = sscanf(lenstring(seps(2)+1:end),'%f');

nsec = hour*3600+minute*60+second;

if ~exist('LEN','var')
    LEN = 200;
end

chunks = linspace(0,nsec,LEN);
bg = cell(LEN,1);

if exist('func','var')
  for i = 1:LEN
    bg{i} = @()(func(get_movie_frame(mname,get_movie_string(chunks(i)))));
  end
else  
  for i = 1:LEN
    bg{i} = @()(get_movie_frame(mname,get_movie_string(chunks(i))));
  end
end

function str = get_movie_string(index)
hours = floor(index/(3600));
minutes = floor(index/60);
seconds = index - 3600*hours-60*minutes;
str = sprintf('%02d:%02d:%02.3f',hours,minutes,seconds);

