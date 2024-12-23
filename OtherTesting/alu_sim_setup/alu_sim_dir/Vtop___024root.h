// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtop.h for the primary calling header

#ifndef VERILATED_VTOP___024ROOT_H_
#define VERILATED_VTOP___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_threads.h"


class Vtop__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtop___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    IData/*31:0*/ RISCALU__DOT__d;
    VL_OUT(d,31,0);
    VL_IN8(clk,0,0);
    CData/*0:0*/ RISCALU__DOT__clk;
    CData/*2:0*/ RISCALU__DOT__funct3;
    CData/*6:0*/ RISCALU__DOT__funct7;
    IData/*31:0*/ RISCALU__DOT__s1;
    IData/*31:0*/ RISCALU__DOT__s2;
    VL_IN8(funct3,2,0);
    VL_IN8(funct7,6,0);
    VL_IN(s1,31,0);
    VL_IN(s2,31,0);
    CData/*0:0*/ RISCALU__DOT__zero;
    VL_OUT8(zero,0,0);
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __VactContinue;
    IData/*31:0*/ __VstlIterCount;
    IData/*31:0*/ __VicoIterCount;
    IData/*31:0*/ __VactIterCount;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;
    VlMTaskVertex __Vm_mtaskstate_7;
    VlMTaskVertex __Vm_mtaskstate_final__nba;

    // INTERNAL VARIABLES
    Vtop__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtop___024root(Vtop__Syms* symsp, const char* v__name);
    ~Vtop___024root();
    VL_UNCOPYABLE(Vtop___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
