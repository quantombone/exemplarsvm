function xxx = replica_hits(I, sbin, target_id, hg_size, NNN)
%% Given an image, an sbin, and the location of the detection window
%% inside target_id, create NNN wiggles by perturbing image content
%% inside the frame

%% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('NNN','var')
  NNN = 100;
end

I = im2double(I);

NEWPAD = ceil(max(hg_size(1),hg_size(2))/2);

u = target_id.offset(1)+NEWPAD;
v = target_id.offset(2)+NEWPAD;

lilI = resize(I,target_id.scale);
f = features(lilI,sbin);
f = padarray(f,[NEWPAD NEWPAD 0]);
xxx_base = f(u-2+(1:hg_size(1)),v-2+(1:hg_size(2)),:);

%%% NOW DO WIGGLES WITHOUT A SLIDE!
xxx = zeros(prod(hg_size),NNN);
xxx(:,1) = xxx_base(:);
for iii = 2:NNN
  %fprintf(1,'$');

  %Add random noise
  curI = lilI + .02*randn(size(lilI));
  
  %Perform random gaussian smoothing
  randsigma = .1+rand;
  curI = (imfilter(curI,fspecial('gaussian',[9 9],randsigma)));
  
  %Perform random shift
  cx = floor(3*rand)-1;
  cy = floor(3*rand)-1;
  curI = circshift2(curI,[cx cy]);

  % rrr = ceil(rand*4);
  % if rrr == 1
  %   TOP = floor(rand*size(curI,1)*.3);
  %   curI(1:TOP,:,:) = 0;
  % elseif rrr == 2
  %   BOT = floor(rand*size(curI,1)*.3);
  %   curI(end-BOT:end,:,:) = 0;
  % elseif rrr == 3
  %   LEFT = floor(rand*size(curI,2)*.3);
  %   curI(:,1:LEFT,:) = 0;
  % else
  %   RIGHT = floor(rand*size(curI,2)*.3);
  %   curI(:,end-RIGHT:end,:) = 0;
  % end
  
  f = features(curI,sbin);
  f = padarray(f,[NEWPAD NEWPAD 0]);
  x = f(u-2+(1:hg_size(1)),v-2+(1:hg_size(2)),:);
  xxx(:,iii) = x(:);
  if 0
    figure(444)
    clf
    imagesc(max(0.0,min(1.0,curI)))
    drawnow
    pause(.2)
  end
end

% return

% clear sg;
% cxs = 1:5:16;
% cys = 1:5:16;
% xxx = zeros(prod(size(w)),0);
% for cxi = 1:length(cxs)
%   for cyj = 1:length(cys)
%     cx = cxs(cxi);
%     cy = cys(cyj);
    
%     curI = I(cx:end,cy:end,:);
    
%     if rand>.5
%       curI = rgb2gray(curI);
%       curI = repmat(curI,[1 1 3]);
%     end
%     %curI = curI + .3*randn(size(curI));
%     %curI = max(0.0,min(1.0,curI));
%     randsigma = .1+rand*2;
%     curI = (imfilter(curI,fspecial('gaussian',[9 9],randsigma)));
%     curI = min(max(curI,0.0),1.0);
    
%     [rs,t] = localizemeHOG(curI,...
%                            sbin,...
%                            {w},...
%                            {b},...
%                            -1.0,1,10,1);
%     if 0
%     mask = zeros(size(w,1),size(w,2));
%     [coarse_boxes] = extract_bbs_from_rs(rs, curI, ...
%                                           sbin, mask);
%     bbb = coarse_boxes(1,:);
%     (bbb(3)-bbb(1)+1) / (bbb(4)-bbb(2)+1)
%     figure(44)
%     clf
%     imagesc(curI)
%     plot_bbox(coarse_boxes(1,:))
%     pause
%     end
%     fprintf(1,'\n');
%     maxer=max(rs.score_grid{1});
%     sg(cxi,cyj) = maxer;
%     xxx(:,end+1) = rs.support_grid{1}{1};
%   end
% end
