function bg = get_movie_bg(mname,LEN)
%Get a movie filename into a virtual 'bg' dataset
  
if ~exist('mname','var') | length(mname)==0
    mname='~/Movies/Caddyshack/Caddyshack.avi';
    mname = '~/Movies/National.Geographic.Clash.of.the.Continents.PDTV.XviD.MP3.MVGroup.org.avi';
end


lenstring = sprintf(['/nfs/baikal/tmalisie/ffmpeg/ffmpeg-0.6/ffmpeg -i' ...
                    ' %s 2>&1 | grep Duration'], mname);
[tmp,lenstring] = unix(lenstring);

commas = find(lenstring==',');
commas = commas(1);
f = strfind(lenstring,'Duration:');
lenstring = lenstring(f+10:commas-1);
seps = find(lenstring==':');
hour = sscanf(lenstring(1:seps(1)-1),'%d');
minute = sscanf(lenstring(seps(1)+1:seps(2)-1),'%d');
second = sscanf(lenstring(seps(2)+1:end),'%f');

nsec = hour*3600+minute*60+second;


% istring = ...
%     sprintf(['/nfs/baikal/tmalisie/ffmpeg/ffmpeg-0.6/ffmpeg' ...
%              '-ss %s -vframes 1  -i %s /tmp/I-%%08d' '.png'],...
%             time, movie);

if ~exist('LEN','var')
    LEN = 200;
end

chunks = linspace(0,nsec,LEN);
bg = cell(LEN,1);

for i = 1:LEN
    bg{i} = @()(get_movie_frame(mname,get_movie_string(chunks(i))));
end

function str = get_movie_string(index)
hours = floor(index/(3600));
minutes = floor(index/60);
seconds = index - 3600*hours-60*minutes;
str = sprintf('%02d:%02d:%02.3f',hours,minutes,seconds);

