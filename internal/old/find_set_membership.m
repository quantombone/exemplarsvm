function [negatives,vals,pos,test,indicator] = find_set_membership(svids,cls)
%Given the detection objects svids and a class cls,
%parse each detection's curid to see which image set it belongs to.
%There are 4 possible sets for a given category: train-, val-, trainval+, test
% Tomasz Malisiewicz (tomasz@cmu.edu)

bgtrain = get_pascal_bg('train',['-' cls]);
bgval = get_pascal_bg('val',['-' cls]);
fg = get_pascal_bg('trainval',cls);
fgtest = get_pascal_bg('test');


bgtrain = cellfun(@(x)get_file_id(x),bgtrain);
bgval = cellfun(@(x)get_file_id(x),bgval);
fg = cellfun(@(x)get_file_id(x),fg);
fgtest = cellfun(@(x)get_file_id(x),fgtest);

ids = cellfun(@(x)x.curid,svids);

negatives = find(ismember(ids,bgtrain));
vals = find(ismember(ids,bgval));
pos = find(ismember(ids,fg));
test = find(ismember(ids,fgtest));
indicator = zeros(length(svids),1);
indicator(negatives) = 1;
indicator(vals) = 2;
indicator(pos) = 3;
indicator(test) = 4;

%left out images must be the difficult in-class images
indicator(indicator==0) = 3;
pos = find(indicator==3);

fprintf(1,'Sets NEG,VAL,POS,TEST=%d,%d,%d,%d\n',...
        length(negatives),length(vals),...
        length(pos),length(test));

%should not get here
if sum(indicator==0)
  fprintf(1,'WARNING FOUND IDS NOT IN THESE SETS\n');
end
