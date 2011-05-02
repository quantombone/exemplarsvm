function pool_loop

for i = 1:1000
  fprintf(1,'iteration %d\n',i);

  %Run training procedure
  pool_train;
 
  BASEDIR = '/nfs/baikal/tmalisie/pool/pool/';
  [a,b] = unix(['rm ' BASEDIR '*mat']);
  
  while 1
    files = dir([BASEDIR '*mat']);
    if length(files) == 200
      break;
    else
      fprintf(1,'Found %d files\n',length(files));
      pause(5)
    end
  end
  
  
end