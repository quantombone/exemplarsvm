function objects = extract_objects(objects,cls)
good_objects = find(( ismember({objects.class},cls) ));
objects = objects(good_objects);

