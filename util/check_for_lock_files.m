function lockfiles = check_for_lock_files(results_directory)
%Uses the unix find command to see if there are any lock files
%(directories which end in ".lock") inside the directory specified
%by results_directory
%
%Returns a cell array of files, potentially of length 0
%Tomasz Malisiewicz (tomasz@csail.mit.edu)

[a,b] = unix(sprintf('find %s -name "*.lock"',results_directory));
if a==1 || length(b) == 0
  lockfiles = {};
else
  b=textscan(b,'%s');
  lockfiles = b{1};
end