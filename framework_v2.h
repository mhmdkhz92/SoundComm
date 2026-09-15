#ifndef SOUNDLINK_FRAMEWORK_H
#define SOUNDLINK_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <complex>
#include <math.h>


namespace soundlink{
    void fatal(const char *s){
        perror(s);exit(1);
    }
    void fail(const char *s){
        fprintf(stderr, "** %s\n", s);exit(1);
    }
    //////////////////////////////////////////////////////////////////////
    // DSP framework
    //////////////////////////////////////////////////////////////////////
    
    // [buff] defines the storage type, scalar or complex 
    // [pipebuf] is a FIFO buffer with multiple readers.
    // [pipewriter] is a client-side hook for writing into a [pipebuf].
    // [pipereader] is a client-side hook reading from a [pipebuf].
    // [runnable] is anything that moves data between [pipebufs].
    // [scheduler] is a global context which invokes [runnables] until fixpoint.
    static const int MAX_PIPES = 64;
    static const int MAX_RUNNABLES = 64;
    static const int MAX_READERS = 8;

    struct pipebuf_common{
        virtual int sizeofT() = 0;
        virtual long long hash() = 0;
        const char *name;
        pipebuf_common(const char *_name) : name(_name){}
    };

    struct runnable_common{
        const char *name;
        runnable_common(const char *_name) : name(_name){}
        virtual void run(){};
        virtual void shutdown(){};
    };

    struct scheduler{
        pipebuf_common *pipes[MAX_PIPES];
        int npipes;
        runnable_common *runnables[MAX_RUNNABLES];
        int nrunnables;
        scheduler(): npipes(0), nrunnables(0){}

        void add_pipe(pipebuf_common *p) {
            if (npipes == MAX_PIPES )
                fail("MAX_PIPES");
            pipes[npipes++] = p;
        }
        void add_runnable(runnable_common *r){
            if ( nrunnables == MAX_RUNNABLES )
                fail("MAX_RUNNABLES");
            runnables[nrunnables++] = r;
        }

        void step() {
            for(int i = 0; i < nrunnables; ++i)
                runnables[i]->run();
        }
        void run(){
            unsigned long long prev_hash = 0;
            while (1){
                step();
                unsigned long long h = hash();
                if ( h == prev_hash ) break;
                prev_hash = h;
            }
        }

        void shutdown() {
        for ( int i=0; i<nrunnables; ++i )
        runnables[i]->shutdown();
        }
        unsigned long long hash(){
        unsigned long long h = 0;
        for ( int i=0; i<npipes; ++i )
            h += (1+i)*pipes[i]->hash();
        return h;
        }
    };
    
    struct runnable:runnable_common {
        runnable(scheduler *_sch, const char *name):runnable_common(name), sch(_sch) {
        sch->add_runnable(this);
        }
    protected:
        scheduler *sch;
    };


    template<typename T>
    struct complex_view{
        T* re;
        T* im;
    };

    template<typename T>
    struct scalar_storage{
        T* buf;
        scalar_storage(int size):buf(new T[size]){};
        void pack(int rd, int bytes){
            memmove(buf, buf + rd, bytes);
        }
        T* write_pointer(int wr){
            return buf + wr;
        }
        T* read_pointer(int rd){
            return buf + rd;
        }
        void write(int offset, const T& value) {
            buf[offset] = value;
        }
        
        ~scalar_storage(){
            delete[] buf;
        }
    };
    template<typename T>
    struct complex_storage{
        T* real;
        T* imag;
        complex_storage(int size){
            real = new T[size];
            imag = new T[size];
        }
        void pack(int rd, int bytes){
            memmove(real, real + rd, bytes);
            memmove(imag, imag + rd, bytes);
        }
        complex_view<T> write_pointer(int wr){
            return{real + wr, imag + wr};
        }
        complex_view<T> read_pointer(int rd){
            return{real + rd, imag + rd};
        }
        void write(int offset, const std::complex<T>& value) {
            real[offset] = value.real();
            imag[offset] = value.imag();
        }
        ~complex_storage(){
            delete[] real;
            delete[] imag;
        }
    };

    template<template<typename> class buf_t, typename T>
    struct pipebuf:pipebuf_common{
        buf_t<T> buf;
        int rds[MAX_READERS];
        int nrd;
        int wr;
        int end;
        int sizeofT(){
            return sizeof(T);
        }
        pipebuf(scheduler *sch, const char *name, unsigned long size): pipebuf_common(name),
        buf(size), nrd(0), wr(0), end(size), min_write(1),
        total_written(0), total_read(0){
            sch->add_pipe(this);
        }
        int add_reader() {
            if ( nrd == MAX_READERS )
                fail("too many readers");
            rds[nrd] = wr;
            return nrd++;
        }
        void pack(){
            int rd = wr;
            for (int i = 0; i < nrd; ++i) 
                if ( rds[i] < rd )
                    rd = rds[i];
            buf.pack(rd, (wr-rd)*sizeof(T));
            wr -= rd;
            for ( int i=0; i<nrd; ++i )
                rds[i] -= rd;
        }
        auto write_pointer(){
            return buf.write_pointer(wr);
        }
        auto read_pointer(int rd){
            return buf.read_pointer(rd);
        }
        void write(const T &e){
            buf.write(wr, e);
        }
        long long hash(){
            return total_written + total_read;
        }
        unsigned long min_write;
        unsigned long total_written, total_read;
    };

    template<template<typename> class buf_t, typename T>
    struct pipewriter{
        pipebuf<buf_t, T>& buf;
        pipewriter(pipebuf<buf_t, T> &_buf, unsigned long min_write=1):buf(_buf) {
            if ( min_write > buf.min_write )
                buf.min_write = min_write;
            }
        // Return number of items writable at this->wr, 0 if full.
        unsigned long writable() {
            if (buf.end-buf.wr < buf.min_write)
                buf.pack();
            return buf.end - buf.wr;
        }
        auto wr(){
            return buf.write_pointer();
        }
        void written(unsigned long n){
            if( buf.wr+n > buf.end ) {
                fprintf(stderr, "Bug: overflow to %s\n", buf.name);
                exit(1);
            }
            buf.wr += n;
            buf.total_written += n;
        }
        inline void write(const T &e) {
            buf.write(e);
            written(1);
        }
    };

    template<template<typename> class buf_t, typename T>
    struct pipereader{
        pipebuf<buf_t, T>& buf;
        int id;
        pipereader(pipebuf<buf_t, T> &_buf):buf(_buf), id(_buf.add_reader()){}
        unsigned long readable(){
            return buf.wr - buf.rds[id];
        }
        auto rd(){
            return buf.read_pointer(rds[id]);
        }
        void read(unsigned long n) {
            if ( buf.rds[id]+n > buf.wr ) {
	            fprintf(stderr, "Bug: underflow from %s\n", buf.name);
	        exit(1);
            }
            buf.rds[id] += n;
            buf.total_read += n;
        }
    };
    typedef uint32_t u32;
    typedef uint16_t u16;
    typedef uint8_t u8;
    typedef std::complex<float> cf32;
}

#endif