// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtop__Syms.h"
#include "Vtop.h"
#include "Vtop___024root.h"

// FUNCTIONS
Vtop__Syms::~Vtop__Syms()
{

    // Tear down scope hierarchy
    __Vhier.remove(0, &__Vscope_RISCALU);

}

Vtop__Syms::Vtop__Syms(VerilatedContext* contextp, const char* namep, Vtop* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    , __Vm_threadPoolp{static_cast<VlThreadPool*>(contextp->threadPoolp())}
    // Setup module instances
    , TOP{this, namep}
{
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-9);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    // Setup scopes
    __Vscope_RISCALU.configure(this, name(), "RISCALU", "RISCALU", -9, VerilatedScope::SCOPE_MODULE);
    __Vscope_TOP.configure(this, name(), "TOP", "TOP", 0, VerilatedScope::SCOPE_OTHER);

    // Set up scope hierarchy
    __Vhier.add(0, &__Vscope_RISCALU);

    // Setup export functions
    for (int __Vfinal = 0; __Vfinal < 2; ++__Vfinal) {
        __Vscope_RISCALU.varInsert(__Vfinal,"clk", &(TOP.RISCALU__DOT__clk), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_RISCALU.varInsert(__Vfinal,"d", &(TOP.RISCALU__DOT__d), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope_RISCALU.varInsert(__Vfinal,"funct3", &(TOP.RISCALU__DOT__funct3), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,2,0);
        __Vscope_RISCALU.varInsert(__Vfinal,"funct7", &(TOP.RISCALU__DOT__funct7), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,1 ,6,0);
        __Vscope_RISCALU.varInsert(__Vfinal,"s1", &(TOP.RISCALU__DOT__s1), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope_RISCALU.varInsert(__Vfinal,"s2", &(TOP.RISCALU__DOT__s2), false, VLVT_UINT32,VLVD_NODIR|VLVF_PUB_RW,1 ,31,0);
        __Vscope_RISCALU.varInsert(__Vfinal,"zero", &(TOP.RISCALU__DOT__zero), false, VLVT_UINT8,VLVD_NODIR|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"clk", &(TOP.clk), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,0);
        __Vscope_TOP.varInsert(__Vfinal,"d", &(TOP.d), false, VLVT_UINT32,VLVD_OUT|VLVF_PUB_RW,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"funct3", &(TOP.funct3), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,1 ,2,0);
        __Vscope_TOP.varInsert(__Vfinal,"funct7", &(TOP.funct7), false, VLVT_UINT8,VLVD_IN|VLVF_PUB_RW,1 ,6,0);
        __Vscope_TOP.varInsert(__Vfinal,"s1", &(TOP.s1), false, VLVT_UINT32,VLVD_IN|VLVF_PUB_RW,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"s2", &(TOP.s2), false, VLVT_UINT32,VLVD_IN|VLVF_PUB_RW,1 ,31,0);
        __Vscope_TOP.varInsert(__Vfinal,"zero", &(TOP.zero), false, VLVT_UINT8,VLVD_OUT|VLVF_PUB_RW,0);
    }
}
