function m = try_reshape(m,cb,N)

% if ~isfield(m,'bg_string1')
%   bg = get_pascal_bg('trainval');
% else
%   bg = get_pascal_bg(m.bg_string1, m.bg_string2);
% end

%bg = cat(1,get_pascal_bg('trainval'),get_pascal_bg('test'));

[aa,bb] = sort(cb(:,end),'descend');

if ~exist('N','var')
  N = 100;
end
xs = zeros(8*8*31,N);
target_id = cell(1,N);
VOCinit;
tic
for i = 1:N
  fprintf(1,'.');
  target_id{i}.level = -1; %% not using this field
  target_id{i}.scale = cb(bb(i),8);
  target_id{i}.offset = cb(bb(i),9:10);
  target_id{i}.flip = cb(bb(i),7);
  target_id{i}.bb = cb(bb(i),1:4);
  target_id{i}.curid = cb(bb(i),11);
  
  I = convert_to_I(sprintf(VOCopts.imgpath,...
                           sprintf('%06d',cb(bb(i),11))));

  
  if (target_id{i}.flip == 1)
    I = flip_image(I);
  end

  I = resize(I,target_id{i}.scale);
  full = features(I,8);
  f = padarray(full,[5 5 0]);

  f = f(target_id{i}.offset(1)+5+(0:7)-1,...
        target_id{i}.offset(2)+5+(0:7)-1,:);

  xs(:,i) = f(:);
  %imagesc(HOGpicture(f))
  %differ=(m.model.w(:)'*f(:) - m.model.b) - cb(bb(i),end);

  %title(num2str(differ))
  %pause

  

  %cb(bb(i),end)
  % keyboard
end
toc


m.model.nsv = cat(2,m.model.nsv,xs);
m.model.svids = cat(2,m.model.svids,target_id);