function I = get_movie_frame(movie, time)
%Extract a frame from a movie using ffmpeg (which must be already
%installed)
% movie: input movie location
% time:  a time in 00:00:00 format
% RELATED FUNCTION: get_movie_bg

%The location of ffmpeg (can be custom if not in $PATH)
FFMPEG_PATH = 'ffmpeg';

%Create a random string from movie filename and time
randstring = [movie '/' time];
[tmp,randstring]=unix(sprintf('md5 -qs %s',randstring));
randstring = randstring(1:end-1);

istring=sprintf([FFMPEG_PATH ' -ss %s -vframes 1  -i %s ' ...
                    ' /tmp/I-%s-%%08d' '.png'],time,movie,randstring);
Ifile = sprintf('/tmp/I-%s-00000001.png',randstring);

%Call ffmpeg to extract the target frame and place into temporary directory
[status,filer] = unix(istring);

%Read the extracted frame
I = imread(Ifile);

%Delete the extracted frame
delete(Ifile);
