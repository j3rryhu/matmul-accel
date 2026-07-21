#include <verilated.h>          // Defines common routines
#include "Vctrl_rf_rf.h"               // From Verilating "sig_tbtop.v"
#include "verilated_vcd_c.h"
#include <iostream>

#define STR(x) #x
#define ST(x) STR(x)
#define CHECK_EQUAL(VAL, EXP) \
    if (EXP!=VAL) printf("Error: %s(0x%0x) != %s(0x%0x)\n", ST(VAL), VAL, ST(EXP), EXP); \
    assert(EXP==VAL);

#define RANGE(NAME, WIDTH, SHIFT) (WIDTH==32 ? NAME : ((NAME >> SHIFT) & ((1 << WIDTH)-1)))

#define HW_WRITE(NAME, IDX, VAL) \
            top->NAME##_wdata IDX = VAL; \
            cycle(top);

#define HW_WRITE_WE(NAME, IDX, VAL) \
            top->NAME##_we IDX = 1; \
            HW_WRITE(NAME, IDX, VAL) \
            top->NAME##_we IDX = 0;

#define SW_READ(ADDR) \
            top->valid = 1; \
            top->read = 1; \
            top->addr = ADDR; \
            top->eval(); \
            rdata = top->rdata; \
            cycle(top); \
            top->valid = 0;

#define SW_WRITE(ADDR, DATA) \
            top->valid = 1; \
            top->read = 0; \
            top->addr = ADDR; \
            top->wdata = DATA; \
            top->wmask = -1; \
            cycle(top); \
            top->valid = 0;
 
Vctrl_rf_rf *top;                      // Instantiation of module

unsigned int main_time = 0;     // Current simulation time

double sc_time_stamp () {       // Called by $time in Verilog
    return main_time;
}

void cycle(Vctrl_rf_rf *top) {
    top->clk = 0;
    top->eval();
    main_time++;
    top->clk = 1;
    top->eval();
    main_time++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);   // Remember args
    Verilated::traceEverOn(true);
    int rdata;

    top = new Vctrl_rf_rf;             // Create instance

    top->resetn = 0;           // Set some inputs
    top->clk = 0;
    top->valid = 0;

    cycle(top);
    cycle(top);
    cycle(top);
    top->resetn = 1;
    std::cout << main_time << ": Remove reset\n";
    cycle(top);
    cycle(top);
    cycle(top);
    cycle(top);
    std::cout << main_time << ": Testcase (CONTROL_start ):\n";
    std::cout << main_time << ": \tSoftware write test\n";
    for (int IDX = 0; IDX <= 0; ++IDX) {
        int temp, value;

        temp = top->CONTROL_start_q << 0;
        temp = (1 << IDX);
        value = 0;

        SW_WRITE( 0, (1 << IDX) )
        CHECK_EQUAL(top->CONTROL_start_q, RANGE(temp, 1, 0))

        SW_WRITE( 0, 0 )
        CHECK_EQUAL(top->CONTROL_start_q, RANGE(value, 1, 0))

    }
    cycle(top);
    cycle(top);
    std::cout << main_time << ": Testcase (CONTROL_input_ready ):\n";
    std::cout << main_time << ": \tSoftware write test\n";
    for (int IDX = 1; IDX <= 1; ++IDX) {
        int temp, value;

        temp = top->CONTROL_input_ready_q << 1;
        temp = (1 << IDX);
        value = 0;

        SW_WRITE( 0, (1 << IDX) )
        CHECK_EQUAL(top->CONTROL_input_ready_q, RANGE(temp, 1, 1))

        SW_WRITE( 0, 0 )
        CHECK_EQUAL(top->CONTROL_input_ready_q, RANGE(value, 1, 1))

    }
    cycle(top);
    cycle(top);
    std::cout << main_time << ": Testcase (CONTROL_weight_ready ):\n";
    std::cout << main_time << ": \tSoftware write test\n";
    for (int IDX = 2; IDX <= 2; ++IDX) {
        int temp, value;

        temp = top->CONTROL_weight_ready_q << 2;
        temp = (1 << IDX);
        value = 0;

        SW_WRITE( 0, (1 << IDX) )
        CHECK_EQUAL(top->CONTROL_weight_ready_q, RANGE(temp, 1, 2))

        SW_WRITE( 0, 0 )
        CHECK_EQUAL(top->CONTROL_weight_ready_q, RANGE(value, 1, 2))

    }
    cycle(top);
    cycle(top);
    std::cout << main_time << ": Testcase (STATUS_out_ready ):\n";
    std::cout << main_time << ": \tHardware write test\n";
    for (int IDX = 0; IDX <= 0; ++IDX) {

        HW_WRITE_WE( STATUS_out_ready, , (1 << (IDX-0)) )
        CHECK_EQUAL(top->ctrl_rf_rf__DOT__STATUS_out_ready_q, (1 << (IDX-0)))

        HW_WRITE_WE( STATUS_out_ready, , 0 )
        CHECK_EQUAL(top->ctrl_rf_rf__DOT__STATUS_out_ready_q, 0)
    }
    std::cout << main_time << ": \tSoftware read test\n";
    for (int IDX = 0; IDX <= 0; ++IDX) {

        top->ctrl_rf_rf__DOT__STATUS_out_ready_q = (1 << (IDX-0));
        HW_WRITE( STATUS_out_ready, , (1 << (IDX-0)) )
        cycle(top);
        SW_READ( 4 )
        CHECK_EQUAL(RANGE(rdata, 1, 0), (1 << (IDX-0)))
    }
    cycle(top);
    cycle(top);
    std::cout << main_time << ": Testcase (STATUS_out_count ):\n";
    std::cout << main_time << ": \tHardware write test\n";
    for (int IDX = 1; IDX <= 16; ++IDX) {

        HW_WRITE_WE( STATUS_out_count, , (1 << (IDX-1)) )
        CHECK_EQUAL(top->ctrl_rf_rf__DOT__STATUS_out_count_q, (1 << (IDX-1)))

        HW_WRITE_WE( STATUS_out_count, , 0 )
        CHECK_EQUAL(top->ctrl_rf_rf__DOT__STATUS_out_count_q, 0)
    }
    std::cout << main_time << ": \tSoftware read test\n";
    for (int IDX = 1; IDX <= 16; ++IDX) {

        top->ctrl_rf_rf__DOT__STATUS_out_count_q = (1 << (IDX-1));
        HW_WRITE( STATUS_out_count, , (1 << (IDX-1)) )
        cycle(top);
        SW_READ( 4 )
        CHECK_EQUAL(RANGE(rdata, 16, 1), (1 << (IDX-1)))
    }

    cycle(top);
    cycle(top);
    std::cout << main_time << ": Test Complete!\n";


    top->final();               // Done simulating
}