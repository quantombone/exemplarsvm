function [negatives,vals,pos] = find_set_membership(m)
%Given the objects inside m.svids, parse their curids to see which
%of the following three sets they belong to

% if ~isfield(m,'bg_string1')
%   bg = get_pascal_bg('trainval');
% else
%   bg = get_pascal_bg(m.bg_string1, m.bg_string2);
% end

bgtrain = get_pascal_bg('train',['-' m.cls]);
bgval = get_pascal_bg('val',['-' m.cls]);
fg = get_pascal_bg('trainval',m.cls);

bgtrain = cellfun(@(x)get_file_id(x),bgtrain);
bgval = cellfun(@(x)get_file_id(x),bgval);
fg = cellfun(@(x)get_file_id(x),fg);

% bgids = zeros(length(bg),1);
% aa = find(ismember(bg,bgtrain));
% bgids(aa) = 1;
% aa = find(ismember(bg,bgval));
% bgids(aa) = 2;
% aa = find(ismember(bg,fg));
% bgids(aa) = 3;

%% HERE we take all string curids, and treat them as literals into
%the images

% s = cellfun(@(x)isstr(x.curid),m.model.svids);
% s = find(s);
% if length(s) > 0
%   train_curids = cell(length(bg),1);
%   for i = 1:length(bg)
%     [tmp,train_curids{i},ext] = fileparts(bg{i});
%   end
  
%   test_curids = cellfun2(@(x)x.curid,m.model.svids(s));

%   [aa,bb] = ismember(test_curids,train_curids);
%   for i = 1:length(s)
%     m.model.svids{s(i)}.curid = bb(i);
%   end  
% end

ids = cellfun(@(x)x.curid,m.model.svids);
%Train only with negatives
negatives = find(ismember(ids,bgtrain));
vals = find(ismember(ids,bgval));
pos = find(ismember(ids,fg));

fprintf(1,'Sets NEG,VAL,POS=%d,%d,%d\n',...
        length(negatives),length(vals),length(pos));


