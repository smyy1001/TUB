#include <sys/wait.h>

#define BUFFER_SIZE 1024
#define PIDS_SIZE 5

/* size of buffer to store user input */
const int input_size = BUFFER_SIZE;

/* amount of child processes */
const int children_amount = PIDS_SIZE;

/* function declarations */
void clear_buffer();
void execute_cmd(char **, int);
void shell_exit();
int update_children();
char ** parse_cmd(int *);
