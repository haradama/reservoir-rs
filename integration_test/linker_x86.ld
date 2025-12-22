ENTRY(_start)

SECTIONS {
    /* QEMU loads the kernel at 1M by default for Multiboot */
    . = 1M;

    .text :
    {
        /* Multiboot Header MUST be at the beginning of the file/segment */
        KEEP(*(.multiboot_header))
        *(.text .text.*)
    }

    .rodata :
    {
        *(.rodata .rodata.*)
    }

    .data :
    {
        *(.data .data.*)
    }

    .bss :
    {
        *(.bss .bss.*)
    }
}