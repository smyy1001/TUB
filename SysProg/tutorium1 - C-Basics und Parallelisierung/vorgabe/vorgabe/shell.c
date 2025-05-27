#include "./shell.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "util/linenoise.h"


/* buffer to store user input */
char input[BUFFER_SIZE];
/* array to hold information (pids) about child processes */
pid_t children[PIDS_SIZE];

int main(){
	int bg;
	char ** args;
	linenoiseHistorySetMaxLen(20);

	while(1){

		/* wait for input */
		args = parse_cmd(&bg);

		/* if valid input, execute */
		if(args){
			execute_cmd(args,bg);
		}
		else {
			printf("Invalid input\n");
		}

		/* free args */
		free(args);
	}
}

/* parse commands and allocates memory for it */
char ** parse_cmd(int * bg){

	/* Reading from the input using the linenoise library.
	 * This lib gives us a history, editing features etc */
	char* input_buf = linenoise("user@SPR: ");
	if(input_buf != NULL){
		linenoiseHistoryAdd(input_buf);
		strncpy(input,input_buf,BUFFER_SIZE);
		free(input_buf);
	}
	else{
		return NULL;
	}

	/* Setting up args, argc and bg*/
	int argc = 1;
	char ** args = NULL;
	*bg = 0;

	/* parsing arguments */
	char * next = strtok(input," \n");

	/* check for unparsable input */
	if(!next)return NULL;

	/* check if exiting */
	if(!strcmp(next,"exit")){
		shell_exit();
	}

	/* parse */
	args = malloc(sizeof(char**));
	args[0] = next;
	while( (next = strtok(NULL," \n")) ){
		args = realloc(args,sizeof(char**) * (argc + 1));
		args[argc] = next;
		argc++;
	}

	/* check if meant to be executed in the background */
	if(!strcmp(args[argc - 1],"&") && argc > 1){
		*bg = 1;
		args[argc-1] = NULL;
	}
	/* more space to terminate with null pointer */
	else {
		args = realloc(args,sizeof(char**) * (argc + 1));
		args[argc] = NULL;
	}
	return args;
}

/* --- Platz für eigene Hilfsfunktionen --- */




/* ---------------------------------------- */

/* execute command
 * char *args[]: (arguments)
 * Enthält den Dateinamen des Auszuführenden Programmes und die
 * dazu übergebenen Argumente
 *	args[0]	      --> Dateiname des Programmes
 *	args[1 bis n] --> Argumente
 *	args[n+1]     --> NULL (Nicht wichtig für die Aufgabe)
 * Gleicher Aufgbau wie char * args[] in der main Methode üblich ist
 *
 * int bg: (background)
 * flag ob der Prozess im Hintergrund ausgeführt werden soll oder die Shell
 * auf den Prozess warten soll
 *	0       --> Shell soll auf den Prozess warten
 *	nicht 0 --> Prozess soll im Hintergrund ausgeführt werden
 */
void execute_cmd(char *args[], int bg){

	/* Relevant für Aufgabe 2 */
	/* clean up children and check if space available */
	if(update_children() >= children_amount){
		printf("Too many processes already running\n");
		return;
	}

	/* TODO */

}



/* Diese Funktion soll alle Child Prozesse die in dem Array "pid_t children[]"
 *  gespeichert sind darauf überprüfen, ob diese noch laufen oder bereits
 *  beendet sind. Wenn ein Prozess beendet wurde soll der Eintrag in dem Array
 *  gelöscht werden, so dass dieser wieder zur Verfügung steht.
 *
 *  return value:
 *   Diese Funktion soll die Anzahl der momentan im Hintergrund laufenden
 *   Prozesse zurückgeben. Prozesse die beendet wurden zählen nicht dazu
 */
int update_children(){

	/* TODO */

	return 0;
}

/* Diese Funktion wird aufgerufen, falls das Stichwort "exit" in der Shell
 * eingegeben wird. Diese Funktion beendet die Shell, jedoch soll sie zuerst darauf
 * warten, dass alle Hintergrundprozesse beendet wurden.
 */
void shell_exit(){

	/* TODO */

	printf("Shell exiting\n");
	exit(0);
}
