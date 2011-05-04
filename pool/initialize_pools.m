function initialize_pools
PPN = 1;
NPROC = 200;

spawn_job('start_pooler',NPROC,PPN);
