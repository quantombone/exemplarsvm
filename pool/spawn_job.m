function timing = spawn_job(process, NPROC, PPN, PAUSE_TIME)
%% A Generic MapReduce driver for spawning jobs on WARP and waiting
%% until they are finished -- there is no reduce!
%% It is finished when all processes are done
%% Input:
%% process: short name of process such as "ei" for
%% exemplar_initialize or "ave" for "apply_voc_exemplars"
%% NPROC: # of processes to start
%% PPN: # of cores per process
%% PAUSEITME: # of seconds to wait between checking if MapReduce is done
%% Tomasz Malisiewicz (tomasz@cmu.edu)
if ~exist('NPROC','var')
  NPROC = 20;
end

if ~exist('PPN','var')
  PPN = 2;
end

if ~exist('PAUSE_TIME','var')
  PAUSE_TIME = 5;
end

starter = tic;
%% start the CLUSTER processes
[a,b]=unix(sprintf(['ssh warp.hpc1 "cd /nfs/hn22/tmalisie/ddip/segment_scripts/ &&' ...
                    ' ./warp_starter.sh %s %d %d"'], process, NPROC, ...
                   PPN));

%customary pause
pause(PAUSE_TIME)

iter = 1;
%%TODO: change my userid tmalisie to something more generic (so
%%      others can use this stuff on warp!)
while 1
  [a,b]=unix(sprintf(['ssh warp.hpc1 "qstat | grep tmalisie" | awk' ...
                      ' ''{print($2)}'' | grep %s'],process));
  if length(b)==0
    res = [];
  else
    res = textscan(b,'%s');
    res = res{1};
  end
  
  if length(res) == 0 && iter > 1
    fprintf(1,'breaking at iteration %d\n',iter);
    break;
  end
  fprintf(1,'[%03d] Njobs [%s] = %d\n',iter,process,length(res));
  pause(PAUSE_TIME)
  iter = iter + 1;
end

%Get final time
timing = toc(starter);