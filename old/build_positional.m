function build_positional(recs2)
%build positional classifier

for i =1:length(recs2)
  
  if length(recs2(i).objects) == 0
    continue
  end
  hitp = find(ismember({recs2(i).objects.class},{'person'}));
  if hitp == 1
    hito = 2;
  else
    hito = 1;
  end
  keyboard
end
