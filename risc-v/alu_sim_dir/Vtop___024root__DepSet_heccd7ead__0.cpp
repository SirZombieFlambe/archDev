// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "verilated.h"
#include "verilated_dpi.h"

#include "Vtop__Syms.h"
#include "Vtop___024root.h"

VL_INLINE_OPT void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    // Body
    vlSelf->RISCALU__DOT__clk = vlSelf->clk;
    vlSelf->RISCALU__DOT__funct3 = vlSelf->funct3;
    vlSelf->RISCALU__DOT__funct7 = vlSelf->funct7;
    vlSelf->RISCALU__DOT__s1 = vlSelf->s1;
    vlSelf->RISCALU__DOT__s2 = vlSelf->s2;
    vlSelf->d = vlSelf->RISCALU__DOT__d;
    vlSelf->zero = (0U == vlSelf->RISCALU__DOT__d);
    vlSelf->RISCALU__DOT__zero = vlSelf->zero;
}

void Vtop___024root___eval_ico(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void Vtop___024root___eval_act(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_act\n"); );
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__1(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__1\n"); );
    // Init
    IData/*31:0*/ __Vtemp_1;
    // Body
    __Vtemp_1 = (((((((((0U == (IData)(vlSelf->funct3)) 
                        | (1U == (IData)(vlSelf->funct3))) 
                       | (2U == (IData)(vlSelf->funct3))) 
                      | (3U == (IData)(vlSelf->funct3))) 
                     | (4U == (IData)(vlSelf->funct3))) 
                    | (5U == (IData)(vlSelf->funct3))) 
                   | (6U == (IData)(vlSelf->funct3))) 
                  | (7U == (IData)(vlSelf->funct3)))
                  ? ((0U == (IData)(vlSelf->funct3))
                      ? (vlSelf->s1 + vlSelf->s2) : 
                     ((1U == (IData)(vlSelf->funct3))
                       ? ((0x1fU >= vlSelf->s2) ? (vlSelf->s1 
                                                   << vlSelf->s2)
                           : 0U) : ((2U == (IData)(vlSelf->funct3))
                                     ? (VL_LTS_III(32, vlSelf->s1, vlSelf->s2)
                                         ? 1U : 0U)
                                     : ((3U == (IData)(vlSelf->funct3))
                                         ? ((vlSelf->s1 
                                             < vlSelf->s2)
                                             ? 1U : 0U)
                                         : ((4U == (IData)(vlSelf->funct3))
                                             ? (vlSelf->s1 
                                                ^ vlSelf->s2)
                                             : ((5U 
                                                 == (IData)(vlSelf->funct3))
                                                 ? 
                                                ((0x20U 
                                                  == (IData)(vlSelf->funct7))
                                                  ? 
                                                 ((0x1fU 
                                                   >= vlSelf->s2)
                                                   ? 
                                                  (vlSelf->s1 
                                                   >> vlSelf->s2)
                                                   : 0U)
                                                  : 
                                                 ((0x1fU 
                                                   >= vlSelf->s2)
                                                   ? 
                                                  VL_SHIFTRS_III(32,32,32, vlSelf->s1, vlSelf->s2)
                                                   : 
                                                  (- 
                                                   (vlSelf->s1 
                                                    >> 0x1fU))))
                                                 : 
                                                ((6U 
                                                  == (IData)(vlSelf->funct3))
                                                  ? 
                                                 (vlSelf->s1 
                                                  | vlSelf->s2)
                                                  : 
                                                 (vlSelf->s1 
                                                  & vlSelf->s2))))))))
                  : vlSelf->RISCALU__DOT__d);
    vlSelf->RISCALU__DOT__d = __Vtemp_1;
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__3(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__3\n"); );
    // Body
    vlSelf->d = vlSelf->RISCALU__DOT__d;
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__4(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__4\n"); );
    // Body
    vlSelf->zero = (0U == vlSelf->RISCALU__DOT__d);
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__5(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__5\n"); );
    // Body
    vlSelf->RISCALU__DOT__zero = vlSelf->zero;
}

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf);
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
void Vtop___024root___eval_nba(Vtop___024root* vlSelf);

void Vtop___024root___eval(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval\n"); );
    // Init
    CData/*0:0*/ __VicoContinue;
    VlTriggerVec<1> __VpreTriggered;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    vlSelf->__VicoIterCount = 0U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        __VicoContinue = 0U;
        Vtop___024root___eval_triggers__ico(vlSelf);
        if (vlSelf->__VicoTriggered.any()) {
            __VicoContinue = 1U;
            if (VL_UNLIKELY((0x64U < vlSelf->__VicoIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__ico(vlSelf);
#endif
                VL_FATAL_MT("/home/zsnider/archDev/risc-v/alu.v", 3, "", "Input combinational region did not converge.");
            }
            vlSelf->__VicoIterCount = ((IData)(1U) 
                                       + vlSelf->__VicoIterCount);
            Vtop___024root___eval_ico(vlSelf);
        }
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        __VnbaContinue = 0U;
        vlSelf->__VnbaTriggered.clear();
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            vlSelf->__VactContinue = 0U;
            Vtop___024root___eval_triggers__act(vlSelf);
            if (vlSelf->__VactTriggered.any()) {
                vlSelf->__VactContinue = 1U;
                if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                    Vtop___024root___dump_triggers__act(vlSelf);
#endif
                    VL_FATAL_MT("/home/zsnider/archDev/risc-v/alu.v", 3, "", "Active region did not converge.");
                }
                vlSelf->__VactIterCount = ((IData)(1U) 
                                           + vlSelf->__VactIterCount);
                __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
                vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
                Vtop___024root___eval_act(vlSelf);
            }
        }
        if (vlSelf->__VnbaTriggered.any()) {
            __VnbaContinue = 1U;
            if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
                Vtop___024root___dump_triggers__nba(vlSelf);
#endif
                VL_FATAL_MT("/home/zsnider/archDev/risc-v/alu.v", 3, "", "NBA region did not converge.");
            }
            __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
            Vtop___024root___eval_nba(vlSelf);
        }
    }
}

#ifdef VL_DEBUG
void Vtop___024root___eval_debug_assertions(Vtop___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->funct3 & 0xf8U))) {
        Verilated::overWidthError("funct3");}
    if (VL_UNLIKELY((vlSelf->funct7 & 0x80U))) {
        Verilated::overWidthError("funct7");}
}
#endif  // VL_DEBUG
