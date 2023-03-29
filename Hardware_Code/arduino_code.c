#include <avr/io.h>
#include <avr/interrupt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <util/delay.h>
#include <avr/pgmspace.h>

void ser_send(uint8_t chr) {
    // wait for last character to be sent
    while (!(UCSR0A & _BV(UDRE0)));
    // put in new one
    UDR0 = chr;
}

// this table tells us what direction the encoder is heading based on the
// current and previous pin states
// we define positive rotation as: BA: 00 -> 10 -> 11 -> 01 = counterclockwise
// we get a 4 bit value with the current (A B) and previous (a b) 
// values of the encoder formatted like BAba
// the low bit of the action is value to add to err and the high 7 bits are the
// signed value to add to the position
#define ACTION_NOTHING (0) // q_val += 0;   q_err += 0;
#define ACTION_POS (2)     // q_val += 1;   q_err += 0;
#define ACTION_NEG (0xFE)  // q_val += -1;  q_err += 0;
#define ACTION_ERR (1)     // q_val += 0;   q_err += 1;

// for high count encoders, we don't need to know each quarter count. on those,
// we only claim movement when the encoder hits all zeros. this also gives a
// little bit of hysteresis.
// #define HIGH_COUNT_ENCODER // NOTE: probably a broken concept!

#ifdef HIGH_COUNT_ENCODER
#define ACTION_POS_QC (ACTION_NOTHING) // quarter count
#define ACTION_POS_FC (ACTION_POS) // full count
#define ACTION_NEG_QC (ACTION_NOTHING)
#define ACTION_NEG_FC (ACTION_NEG)
#else
#define ACTION_POS_QC (ACTION_POS) // quarter count
#define ACTION_POS_FC (ACTION_POS) // full count
#define ACTION_NEG_QC (ACTION_NEG)
#define ACTION_NEG_FC (ACTION_NEG)
#endif

// BA = new state of B and A lines
// ba = old state of B and A lines
const uint8_t PROGMEM action_table[16] = {
    // BAba == 0 0 0 0 -> no change
    ACTION_NOTHING,
    // BAba == 0 0 0 1 -> A: 1->0 when B=0
    ACTION_POS_FC, // new state is 00, so one full count has elapsed
    // BAba == 0 0 1 0 -> B: 1->0 when A=0
    ACTION_NEG_FC, // new state is 00, so one full count has elapsed
    // BAba == 0 0 1 1 -> both change so we don't know the direction
    ACTION_ERR,

    // BAba == 0 1 0 0 -> A: 0->1 when B=0
    ACTION_NEG_QC,
    // BAba == 0 1 0 1 -> no change
    ACTION_NOTHING,
    // BAba == 0 1 1 0 -> both change so we don't know the direction
    ACTION_ERR,
    // BAba == 0 1 1 1 -> B: 1->0 when A=1
    ACTION_POS_QC,

    // BAba == 1 0 0 0 -> B: 0->1 when A=0
    ACTION_POS_QC,
    // BAba == 1 0 0 1 -> both change so we don't know the direction
    ACTION_ERR,
    // BAba == 1 0 1 0 -> no change
    ACTION_NOTHING,
    // BAba == 1 0 1 1 -> A: 1->0 when B=1
    ACTION_NEG_QC,

    // BAba == 1 1 0 0 -> both change so we don't know the direction
    ACTION_ERR,
    // BAba == 1 1 0 1 -> B: 0->1 when A=1
    ACTION_NEG_QC,
    // BAba == 1 1 1 0 -> A: 0->1 when B=1
    ACTION_POS_QC,
    // BAba == 1 1 1 1 -> no change
    ACTION_NOTHING,
};

// quadrature state
// last value of the quadrature AB lines
uint8_t last = 0;
 // current position
volatile int16_t q_val = 0;
// current error count (i.e. if the encoder moved too fast and we missed it)
volatile uint8_t q_err = 0;

ISR(PCINT0_vect) {
    // get the current values of the pins
    uint8_t curr = PINB;
    // add in the last ones, so we can look up the action
    uint8_t both = (curr & 0x0C) | last;
    // remember what the last values were so we can figure out rotation
    // direction
    last = both >> 2;

    // figure out what to do based on the action table
    uint8_t action = pgm_read_byte(&action_table[both]);
    // low bit is value to add to err
    uint8_t err = action & 1;
    q_err += err;
    // high 7 bits is signed value to add to val. arithmetic shift right so we
    // sign-extend to an 8 bit value. asm saves a few instructions
    asm("asr %0" : "+r" (action));
    q_val += (int8_t)action;
}


volatile uint8_t time_ms = 0;
// keep time. this interrupt has to be very fast since it's taking time from the
// quadrature interrupt. to this end, we just bump the counter and let the main
// loop handle everything else.
ISR(TIMER2_COMPA_vect) {
    time_ms += 1;
}

int main(void) {
    cli();

    // enable UART
    // baudrate 115.2kbps
    UBRR0H = 0;
    UBRR0L = 16;
    UCSR0A = _BV(U2X0);
    // enable serial tx and rx engines
    UCSR0B = _BV(TXEN0) | _BV(RXEN0);
    UCSR0C = 3<<UCSZ00; // set 8 bit characters

    // set up the pin change interrupts to notify us of quadrature happenings
    // A is connected to PB2 (PCINT2) and B is connected to PB3 (PCINT3)
    PCMSK0 = _BV(PCINT2) | _BV(PCINT3);
    PCICR = _BV(PCIE0);

    // set up timer 2 to bother us at 1KHz so we can keep time
    TCCR2A = _BV(WGM21);
    TIMSK2 = _BV(OCIE2A);
    // 128*125 = 16000 / 16MHz = 1/1000 = 1ms period
    OCR2A = 125;
    TCCR2B = _BV(CS22) | _BV(CS20); // clk/128 prescale

    // INA1: PD7
    // INB1: PB0
    // PWM1: PD5
    // EN1: PC0
    // CS1:  PC2

    // INA2: PD4
    // INB2: PB1
    // PWM2: PD6
    // EN2: PC1
    // CS2: PC3 

    // set all the pins to outputs
    DDRC = _BV(0) | _BV(1);
    DDRB = _BV(0) | _BV(1);
    DDRD = _BV(7) | _BV(5) | _BV(4) | _BV(6);

    // enable the drivers
    PORTC |= _BV(0) | _BV(1);
    // set "positive direction": driver 1 outputs + and 2 outputs -
    PORTD |= _BV(7);
    PORTB |= _BV(0);
    PORTD &= ~_BV(4);
    PORTB &= ~_BV(1);

    // set up timer 0 to run motor
    // we use fast PWM mode (all WGM bits set)
    // and set PWM output to non-inverting mode for both output channels
    TCCR0A =  _BV(COM0A1) | _BV(COM0B1) | _BV(WGM01) | _BV(WGM00);
    // run the timer at 16MHz/8 = 2MHz
    // so that wrapping at 256 = ~8KHz switching frequency
    TCCR0B = _BV(CS01);

    OCR0A = 0;
    OCR0B = 0;

    sei();

    // remember how long it took us to receive the last speed command so we can
    // tell the host the current rate
    uint8_t last_time_ms = 0;

    // remember the last commanded speed value. this way we can update it at the
    // start of the loop and have a consistent delay.
    uint8_t last_speed_cmd = 0;

    // remember the error state so we can notify the host and lock out commands
    uint8_t last_error = 1; // just powered on error

    while (1) {
        cli();
        // get current quadrature state. without interrupts running so we get a
        // consistent snapshot.
        uint16_t curr_q = (uint16_t)q_val;
        uint8_t curr_err = q_err;

        sei();

        // update the motor speed with the last command. doing it at the top of
        // the loop ensures that the time between capturing the position and
        // updating the motor is constant.
        if (last_speed_cmd & 0x80) {
            // set "positive direction": driver 1 outputs - and 2 outputs +
            PORTD |= _BV(4);
            PORTB |= _BV(1);
            PORTD &= ~_BV(7);
            PORTB &= ~_BV(0);
        } else {
            // set "positive direction": driver 1 outputs + and 2 outputs -
            PORTD |= _BV(7);
            PORTB |= _BV(0);
            PORTD &= ~_BV(4);
            PORTB &= ~_BV(1);
        }

        uint8_t out = 0;
        if (!last_error)
            out = (uint8_t)((last_speed_cmd & 0x7F) << 1);
        OCR0A = out;
        OCR0B = out;

        if (last_speed_cmd == 0x80) // acknowledge error
            last_error = 0;

        // tell the host the current position and how long the last update took
        // (or the error state if there is one)
        ser_send(curr_q&0xFF);
        ser_send(curr_q>>8);
        ser_send(curr_err);
        if (last_error)
            ser_send(last_error | 0x80);
        else
            ser_send(last_time_ms);

        // wait for a new speed command
        while (!(UCSR0A & _BV(RXC0))) {
            if (time_ms >= 50) {
                // 50ms is way too long for a new command. don't wait any more.
                break;
            }
        }
        // make sure we actually have a command
        if (UCSR0A & _BV(RXC0)) {
            // get it
            last_speed_cmd = UDR0;
            // and remember how long it took
            last_time_ms = time_ms;
            if (time_ms < 20) { // are we in under time? (for 50hz updates)
                // wait until we've hit it
                while (time_ms != 20) {};
                // then zero the timer and continue to the next loop
                time_ms = 0;
            } else { // we're over time by some
                // wait until the time changes (so we know we are starting with
                // a fresh millisecond)
                uint8_t old_time_ms = time_ms;
                while (old_time_ms == time_ms) {};
                // then zero the timer and continue to next loop
                time_ms = 0;
            }
        } else {
            // we didn't so we must have timed out
            if (!last_error) last_error = 2; // timeout error
            last_time_ms = 0xFF;
            // set the motor to 0%
            OCR0A = 0;
            OCR0B = 0;
            // we still have to maintain synchronization so wait until we do get
            // a command
            while (!(UCSR0A & _BV(RXC0)));
            last_speed_cmd = UDR0;
            // but ignore it because it's based on outdated information
            last_speed_cmd = 0;
            // wait until the time changes (so we know we are starting with a
            // fresh millisecond)
            uint8_t old_time_ms = time_ms;
            while (old_time_ms == time_ms) {};
            // then zero the timer and continue to next loop
            time_ms = 0;
        }
    }
}
