function myRandomize

if  exist('RandStream', 'builtin')
    RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)))
else
    try
        rand('twister', sum(100*clock));
    catch
        rand('seed', sum(100*clock));
    end
end
