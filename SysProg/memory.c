#include <stdio.h>
#include <stdlib.h>

#define BOLD "\033[1m"
#define YELLOW "\033[33m"
#define DARK_GREY "\033[90m"
#define RESET "\033[0m"

// compile with gcc -o memory memory.c
// run with ./memory

void clear_screen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void wait_to_continue() {
    printf("%s\nPress enter to continue...%s\n", DARK_GREY, RESET);
    getc(stdin);
    clear_screen();
}

int main() {
    uint8_t arr[9] = { 
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08
    };

    //////////////////////////
    // LESSON 1
    clear_screen();
    printf("%s%s\nLESSON 1%s\n", BOLD, YELLOW, RESET);
    printf("\narr[9] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08} @ %p\n", arr);
    printf("\nCan we cast arr to a pointer?\n\n");
    uint8_t *ptr = arr;
    printf("arr: %p = ptr: %p\n", arr, ptr);

    printf("\nYes! How cool! arr is simply a pointer to the first element of the array!\n");
    printf("Or in other words, arr holds an address, namely the address of the first element of the array.\n");

    wait_to_continue();

    //////////////////////////
    // LESSON 2
    printf("\n%s%sLESSON 2%s\n", BOLD, YELLOW, RESET);
    printf("\narr[9] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08} @ %p\n", arr);

    printf("\nLets access the array using the [] operator (arr[0], arr[1], ..., arr[8])\n");
    printf("[");
    for (int i = 0; i < 9; i++) {
        uint8_t num = ptr[i];
        printf("%d ", arr[i]);
    }
    printf("]\n\n");


    printf("Lets access the array using pointer arithmetic and dereferencing (*(ptr + 0), *(ptr + 1), ..., *(ptr + 8))\n");
    printf("[");
    for (int i = 0; i < 9; i++) {
        uint8_t* tmp = ptr + i; // to get the address of the i-th element => use + operator 
        uint8_t num = *(tmp); // to get the value at the address => use * operator
        printf("%d ", arr[i]);
    }
    printf("]\n\n");
    printf("Hey! We can access the array using the [] operator or pointer arithmetic and dereferencing! They are equivalent!\n");
    printf("But remember, the pointer was of type uint8_t, so it increments the address by 1 byte and reads 1 byte at a time!\n");

    wait_to_continue();

    //////////////////////////
    // LESSON 3
    printf("\n%s%sLESSON 3%s\n", BOLD, YELLOW, RESET);
    printf("\narr[9] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08} @ %p\n", arr);

    printf("\nLets try some other cool stuff with a pointer of type uint16_t\n");
    uint16_t *ptr16bit = (uint16_t *)arr;
    printf("\narr: %p = ptr16bit: %p\n", arr, ptr16bit);
    printf("arr[0] = %d != ptr16bit[0]: %d\n", arr[0], ptr16bit[0]);
    printf("\nWhy is ptr16bit[0] = 265?\nBecause ptr16bit is a pointer to a 16-bit integer, so it reads 2 bytes at a time.\n");
    printf("arr[0] = 0x00 and arr[1] = 0x01, so ptr16bit[0] = 0x0100 = 256!\n");

    printf("\nSo, what is ptr16bit[1]?\n");
    printf("ptr16bit[1] = %d\n", ptr16bit[1]);
    printf("arr[2] = 0x02 and arr[3] = 0x03, so ptr16bit[1] = 0x0302 = 770!\n");

    printf("\nDoes this also work with pointer arithmetic and dereferencing?\n");
    printf("*(ptr16bit + 1) = %d\n", *(ptr16bit + 1));
    printf("Yes, it does! *(ptr16bit + 1) = ptr16bit[1] = 770!\n");

    wait_to_continue();

    //////////////////////////
    // LESSON 4

    printf("\n%s%sLESSON 4%s\n", BOLD, YELLOW, RESET);
     printf("\narr[9] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08} @ %p\n", arr);
    uint8_t *ptr8 = (uint8_t *)arr;
    uint16_t *ptr16  = (uint16_t *)ptr8; // or (uint16_t *)arr
    uint32_t *ptr32 = (uint32_t *)ptr16; // or (uint32_t *)arr
    uint64_t *ptr64 = (uint64_t *)ptr32; // or (uint64_t *)arr

    printf("\nLets print the addresses of the elements of the array using different pointer types:\n\n");
    for (int i = 0; i < 9; i++) {
        printf("(ptr8 + %d) = %p", i, ptr8 + i);
        if (i %2 == 0) {
            printf(" = (ptr16 + %d) = %p", i/2, ptr16 + i/2);
        }

        if (i % 4 == 0) {
            printf(" = (ptr32 + %d) = %p", i/4, ptr32 + i/4);
        }

        if (i % 8 == 0) {
            printf(" = (ptr64 + %d) = %p", i/8, ptr64 + i/8);
        }

        printf("\n");
    }

    printf("\nSo, depending on the pointer type, the address is incremented by 1, 2, 4 or 8 bytes!\n");
    printf("\nThis is cool! I can access the memory in any way I want!\n\n");

    printf("*((uint8_t*) (ptr32 + 1)) = %d\t\t(fifth byte of 9)\n", *((uint8_t *)(ptr32 + 1)));
    printf("*((uint8_t*) (ptr16 + 3)) = %d\t\t(seventh byte of 9)\n", *((uint8_t *)(ptr16 + 3)));
    printf("*((uint8_t*) (ptr64 + 1)) = %d\t\t(ninth byte of 9)\n", *((uint8_t *)(ptr64 + 1)));

    wait_to_continue();

    //////////////////////////
    // LESSON 5
    printf("\n%s%sLESSON 5%s\n", BOLD, YELLOW, RESET);
    printf("\nnum = 0x0706050403020100\n");
    printf("\nLets do the last really cool thing!\n");
    uint64_t num = 0x0706050403020100;

    uint8_t *num_ptr8 = (uint8_t *)&num;        // to get the address of where the data is stored => use & operator
    uint16_t *num_ptr16 = (uint16_t *)num_ptr8; // or (uint16_t *)&num
    uint32_t *num_ptr32 = (uint32_t *)num_ptr16; // or (uint32_t *)&num
    uint64_t *num_ptr64 = (uint64_t *)num_ptr32; // or (uint64_t *)&num
    
    printf("Given some number num = 0x0706050403020100, lets access it byte by byte!\n\n");
    printf("Lets print the bytes of the number using different pointer types\n");
    printf("*((uint8_t*) &num) = num_ptr8[0] = %d\t\t\t\t(first byte)\n", num_ptr8[0]);
    printf("*((uint16_t*) &num) = num_ptr16[0] = %d\t\t\t(first two bytes)\n", num_ptr16[0]);
    printf("*((uint32_t*) &num) = num_ptr32[0] = %d\t\t\t(first four bytes)\n", num_ptr32[0]);
    printf("*((uint64_t*) &num) = num_ptr64[0] = %llu\t\t(first eight bytes)\n", num_ptr64[0]);

    printf("\nIs this what you expected? It should be!\n");
    printf("num_ptr8[0] = 0x00 (=0)\nnum_ptr16[0] = 0x0100 (=256)\nnum_ptr32[0] = 0x03020100 (=50462976)\nnum_ptr64[0] = 0x0706050403020100 (=506097522914230528)!\n");

    printf("\nSo can I also access the 5th byte of the 8-byte number?\n");
    printf("num_ptr8[4] = %d\n", num_ptr8[4]);
    printf("*(num_ptr8 + 4) = %d\n", *(num_ptr8 + 4));
    printf("Yes, I can! num_ptr8[4] = 0x04 (=4)!\n");

    printf("\nAnd what about the 5th and 6th byte of the 8-byte number?\n");
    printf("*((uint16_t *) num_ptr8 + 4) = %d\n", *((uint16_t *)num_ptr8 + 2));
    printf("Yes, I can! *((uint16_t *) num_ptr8 + 4) = 0x0504 (=1284)!\n");

    wait_to_continue();

    //////////////////////////
    // SUMMARY
    printf("\n%s%sSUMMARY%s\n", BOLD, YELLOW, RESET);
    printf("\nCasting is cool! It simply tells the compiler how to interpret the memory!\n");
    printf("\nGiven some address, I can read the memory in any way I want (by casting the pointer to the correct type)!\n");
    printf("Example: If I want to read the next 4 bytes as a 32-bit unsigned integer, I simply cast the pointer to a uint32_t pointer and dereference it!\n");
    printf("\nI can also access any byte I want by using pointer arithmetic (and casting to the correct type beforehand)!\n");
    printf("Example: If I want to read the 7th byte of a 8-byte number, I cast the pointer to a uint8_t pointer and add 6 (zero bases indxing!) to it!\n");
    printf("Example: But I could also cast the pointer to a uint16_t pointer and add 3 to it, and then cast it to a uint8_t pointer and dereference it!\n");
    printf("Example: Or I could cast the pointer to a uint32_t pointer and add 1 to it, cast it to a uint16_t pointer and add 1 to it, and then cast it to a uint8_t pointer and dereference it!\n");
    printf("\nImportant to remember: An array (variable) simply holds a pointer to the first element of the array, exatly like malloc returns a pointer to the first byte of the allocated memory!\n");

    wait_to_continue();
}
