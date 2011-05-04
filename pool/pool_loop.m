function pool_loop

%%make sure directory is clear
%fff = '004365.1.car.mat';

allfiles = dir(['/nfs/baikal/tmalisie/nips11/local/VOC2007/exemplars/*train*mat']);

%allfiles = dir(['/nfs/baikal/tmalisie/nips11/local/VOC2007/exemplars/000540.1.train.mat'])
%allfiles = dir(['/nfs/baikal/tmalisie/nips11/local/VOC2007/dalals/dalal.train.mat']);

for giter = 1:length(allfiles)
  fff = allfiles(giter).name;

  finalfile = ['/nfs/baikal/tmalisie/finalpool/' fff];
  
  lockfile = [finalfile '.lock'];

  if fileexists(finalfile) || mymkdir_dist(lockfile) == 0
    continue
  end
  
  for i = 1:5
    fprintf(1,'iteration %d\n',i);
    
    BASEDIR = '/nfs/baikal/tmalisie/pool/pool/';
    [a,b] = unix(['rm ' BASEDIR '*' fff]);
    
    
    if (i == 1)
      unix(sprintf('cp /nfs/baikal/tmalisie/nips11/local/VOC2007/exemplars/%s /nfs/baikal/tmalisie/pool/',fff));
    end
    while 1
      files = dir([BASEDIR '*' fff]);
      if length(files) == 200
        break;
      else
        fprintf(1,'Found %d files\n',length(files));
        pause(5)
      end
    end
    
    %Run training procedure
    [status,m] = pool_train(fff);
    
    figure(1)
    set(gcf,'PaperPosition',[0 0 10 10]);
    print(gcf,'-dpng',[finalfile '.pic.' num2str(i) '.png'])
    
    figure(2)
    set(gcf,'PaperPosition',[0 0 10 10]);
    print(gcf,'-dpng',[finalfile '.plot.' num2str(i) '.png'])
    
    if status == 1
      save(['/nfs/baikal/tmalisie/finalpool/' fff],'m');
      delete(['/nfs/baikal/tmalisie/pool/' fff]);
      unix(['rm /nfs/baikal/tmalisie/pool/pool/*' fff]);
      break;
    end
  end
  
  if fileexists(lockfile)
    rmdir(lockfile)
  end
end
