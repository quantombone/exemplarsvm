function I = get_movie_frame(movie, time)
%Extract a frame from a movie using ffmpeg (which must be already
%installed)
% move:  input move location
% time:  a time in 00:00:00 format

istring=sprintf(['/nfs/baikal/tmalisie/ffmpeg/ffmpeg-0.6/ffmpeg -ss %s -vframes 1  -i %s ' ...
                    ' /tmp/I-%%08d' '.png'],time,movie);

[status,filer] = unix(istring);
I = imread('/tmp/I-00000001.png');