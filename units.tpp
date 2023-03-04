/**
    UNITS is a library built to maintain some dimensional-type safety to 
    physcs calculation
    @file units.tpp
    @author Marcus DuPont
    @version 0.1 05/10/22
*/
namespace units
{
    template <typename T>
    constexpr int sgn(T val)
    {
        return (T(0) < val) - (val < T(0));
    }
    //=======================================
    // Mass type specializations
    //=======================================

    template<typename P>
    class Kilogram;

    template<typename P>
    class SolarMass;

    template<typename P>
    class Gram
    {
        public:
        Gram(const P initVal) : value(initVal){}
        Gram(SolarMass<P> m) : value(1.988e33 * m.value) {}
        Gram(Kilogram<P> m)  : value(1e3 * m.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& os, const Gram<P>& mass)
    {
        os << mass.value << " [g]";
        return os;
    }

    template<typename P>
    class SolarMass
    {
        public:
        SolarMass(const P initVal) : value(initVal){}
        SolarMass(Gram<P> m) : value(5.02785e-34 * m.value) {}
        SolarMass(Kilogram<P> m)  : value(5.0279e-31 * m.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& os, const SolarMass<P>& mass)
    {
        os << mass.value << " [M_sun]";
        return os;
    }

    template<typename P>
    class Kilogram
    {
        public:
        Kilogram(const P initVal) : value(initVal){}
        Kilogram(SolarMass<P> m) : value(1.988e30* m.value) {}
        Kilogram(Gram<P>  m) : value(1e-3 * m.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& os, const Kilogram<P>& mass)
    {
        os << mass.value << " [kg]";
        return os;
    }

    //=======================================
    // Time type specializations
    //=======================================
    template<typename P>
    class Year;

    template<typename P>
    class Day;

    template<typename P>
    class Hour;

    template<typename P>
    class Second
    {
        public:
        Second(const P initVal) : value(initVal){}
        Second(Year<P> t) : value(31556952.0 * t.value) {}
        Second(Day<P> t) : value(24.0 * 60.0 * t.value) {}
        Second(Hour<P> t) : value(60.0 * t.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& os, const Second<P>& t)
    {
        os << t.value << " [s]";
        return os;
    }

    template<typename P>
    class Year
    {
        public:
        Year(const P initVal) : value(initVal){}
        Year(Second<P> t) : value(t.value * (1.0 / 31556952.0)) {}
        Year(Day<P> t) : value(t.value * (1.0 / 365.0)) {}
        Year(Hour<P> t) : value(t.value * (1.0 / 24.0 / 365.0)) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Year<P>& t)
    {
        out << t.value << " [yr]";
        return out;
    }

    template<typename P>
    class Day
    {
        public:
        Day(const P initVal) : value(initVal){}
        Day(Year<P> t)   : value(365.0 * t.value) {}
        Day(Second<P> t) : value(t.value * (1.0 /(24.0 * 3600.0))) {}
        Day(Hour<P> t)   : value(t.value * (1.0 / 24.0)) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Day<P>& t)
    {
        out << t.value << " [day]";
        return out;
    }

    template<typename P>
    class Hour
    {
        public:
        Hour(const P initVal) : value(initVal){}
        Hour(Year<P> t) : value(31556952.0 * t.value) {}
        Hour(Day<P> t) : value(24.0 * t.value) {}
        Hour(Second<P> t) : value(t.value * (1.0 / 3600.0)) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Hour<P>& t)
    {
        out << t.value << " [hr]";
        return out;
    }

    //=======================================
    // Length type specializations
    //=======================================
    template<typename P>
    class Parsec;

    template<typename P>
    class Lightyear;

    template<typename P>
    class Kilometer;

    template<typename P>
    class Meter;

    template<typename P>
    class Centimeter
    {
        public:
        Centimeter(const P initVal) : value(initVal){}
        Centimeter(Parsec<P> l)    : value(3.086e18 * l.value) {}
        Centimeter(Lightyear<P> l) : value(9.461e17 * l.value) {}
        Centimeter(Kilometer<P> l) : value(1e5      * l.value) {}
        Centimeter(Meter<P> l)     : value(100.0    * l.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Centimeter<P>& ell)
    {
        out << ell.value << " [cm]";
        return out;
    }

    template<typename P>
    class Meter
    {
        public:
        Meter(const P initVal) : value(initVal){}
        Meter(Centimeter<P> l) : value(1e-2 * l.value) {}
        Meter(Kilometer<P> l)  : value(1e3 * l.value) {}
        Meter(Lightyear<P> l)  : value(9.461e15 * l.value) {}
        Meter(Parsec<P> l)     : value(3.086e16 * l.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Meter<P>& ell)
    {
        out << ell.value << " [m]";
        return out;
    }

    template<typename P>
    class Kilometer
    {
        public:
        Kilometer(const P initVal) : value(initVal){}
        Kilometer(Centimeter<P> l) : value(1e-5 * l.value) {}
        Kilometer(Meter<P> l)      : value(1e-3 * l.value) {}
        Kilometer(Lightyear<P> l)  : value(9.461e+12 * l.value) {}
        Kilometer(Parsec<P> l)     : value(3.086e+13 * l.value) {} 
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Kilometer<P>& ell)
    {
        out << ell.value << " [km]";
        return out;
    }
    
    template<typename P>
    class Lightyear
    {
        public:
        Lightyear(const P initVal) : value(initVal){}
        Lightyear(Centimeter<P> l) : value(1.057e-18 * l.value) {}
        Lightyear(Meter<P> l)      : value(1.057e-16 * l.value) {}
        Lightyear(Kilometer<P> l)  : value(1.057e-13 * l.value) {}
        Lightyear(Parsec<P> l)     : value(3.26156   * l.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Lightyear<P>& ell)
    {
        out << ell.value << " [ly]";
        return out;
    }
    template<typename P>
    class Parsec
    {
        public:
        Parsec(const P initVal) : value(initVal){}
        Parsec(Centimeter<P> l) : value( 3.24078e-19 * l.value) {}
        Parsec(Meter<P> l)      : value(3.24078e-17  * l.value) {}
        Parsec(Kilometer<P> l)  : value(3.24078e-14  * l.value) {}
        Parsec(Lightyear<P> l)  : value( 0.306601    * l.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Parsec<P>& ell)
    {
        out << ell.value << " [pc]";
        return out;
    }

    //=======================================
    // Charge type specializations
    //=======================================
    template<typename P>
    class Coulomb;

    template<typename P>
    class StatCoulomb
    {
        public:
        StatCoulomb(const P initVal) : value(initVal){}
        StatCoulomb(Coulomb<P> q) : value(2997924580.0 * q.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const StatCoulomb<P>& ell)
    {
        out << ell.value << " [statC]";
        return out;
    }

    template<typename P>
    class Coulomb
    {
        public:
        Coulomb(const P initVal) : value(initVal){}
        Coulomb(StatCoulomb<P> q) : value(3.33564e-10 * q.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Coulomb<P>& ell)
    {
        out << ell.value << " [C]";
        return out;
    }

    //=======================================
    // Temperature type specializations
    //=======================================
    template<typename P>
    class Celcius;

    template<typename P>
    class Fahrenheit;

    template<typename P>
    class ElectronVolt;

    template<typename P>
    class Kelvin
    {
        public:
        Kelvin(const P initVal) : value(initVal){}
        Kelvin(Fahrenheit<P> x)   : value((32.0 * x.value - 32.0) * (5.0 / 9.0) + 273.15) {}
        Kelvin(Celcius<P> x)      : value(x.value + 273.15) {}
        Kelvin(ElectronVolt<P> x) : value(1.160451812e4 * x.value) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Kelvin<P>& ell)
    {
        out << ell.value << " [K]";
        return out;
    }

    template<typename P>
    class Fahrenheit
    {
        public:
        Fahrenheit(const P initVal) : value(initVal){}
        Fahrenheit(Kelvin<P> x)       : value(((x.value - 273.15) * (5.0 / 9.0) + 32.0) / 32.0) {}
        Fahrenheit(Celcius<P> x)      : value(x.value * (9.0 / 5.0) + 32.0) {}
        Fahrenheit(ElectronVolt<P> x) : value(8.4365061248733e-23 * x.value * (9.0 / 5.0) + 32.0) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Fahrenheit<P>& ell)
    {
        out << ell.value << " [Fah]";
        return out;
    }

    template<typename P>
    class Celcius
    {
        public:
        Celcius(const P initVal) : value(initVal){}
        Celcius(Fahrenheit<P> x)   : value((32.0 * x.value - 32.0) * (5.0 / 9.0)) {}
        Celcius(Kelvin<P> x)      : value(x.value - 273.15) {}
        Celcius(ElectronVolt<P> x) : value(8.4365061248733e-23 * x.value) {}
        // friend std::ostream& operator<<(std::ostream& out, const Celcius<P>& ell);
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Celcius<P>& ell)
    {
        out << ell.value << " [Cel]";
        return out;
    }

    template<typename P>
    class ElectronVolt
    {
        public:
        ElectronVolt(const P initVal) : value(initVal){}
        ElectronVolt(Fahrenheit<P> x)   : value(((32 * x.value - 32.0) * 5.0 / 9.0 + 273.15)*(1.0/1.160451812e4)) {}
        ElectronVolt(Celcius<P> x)      : value((x.value - 273.15) * (1.0/1.160451812e4) ) {}
        ElectronVolt(Kelvin<P> x) : value(x.value * (1.0/1.160451812e4) ) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const ElectronVolt<P>& ell)
    {
        out << ell.value << " [eV]";
        return out;
    }

    //=======================================
    // Irradiance type specializations
    //=======================================
    // Erg per square meter per second
    template<typename P>
    class ErgCM2P2{};

    //=======================================
    // Angle type specializations
    //=======================================
    template<typename P>
    class Degree;

    template<typename P>
    class Radian
    {
        public:
        Radian(const P initVal) : value(initVal){}
        Radian(Degree<P> a) : value(a.value * (M_PI / 180.0)) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Radian<P>& ell)
    {
        out << ell.value << " [rad]";
        return out;
    }

    template<typename P>
    class Degree
    {
        public:
        Degree(const P initVal) : value(initVal){}
        Degree(Radian<P> a) : value(a.value * (180.0 / M_PI)) {}
        P value;
    };

    template<typename P>
    std::ostream& operator<<(std::ostream& out, const Degree<P>& ell)
    {
        out << ell.value << " [deg]";
        return out;
    }

    template<typename T>
    struct yes{
        yes() = default;
        ~yes() {};
        void foo()
        {
            if constexpr(std::is_integral_v<T>)
            {
                std::cout << "yes!" << "\n";
            } else {
                std::cout << "no!" << "\n";
            }
        }
    };
    /*
    Strong typed quantities class that allows conversion from 
    one set of units to another. Useful for dimensional analyis,
    so in a way, we have some ``dimensional safety'' when 
    performing physics calculation
    */
    template<
    typename P,               // Precision
    typename m,               // power of mass
    typename l,               // power of length
    typename t,               // power of time
    typename q,               // power of charge
    typename temp,            // power of temperature
    typename intensity,       // power of irradiance
    typename angle,           // power of angle
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    struct quantity {
        explicit constexpr quantity(P initVal = 0): 
        value(initVal),
        mType(M),
        lType(L),
        tType(T),
        qType(Q),
        tempType(K),
        radType(I),
        aType(A),
        powm(m::num / m::den),
        powt(t::num / t::den),
        powl(l::num / l::den),
        powq(q::num / q::den),
        powi(intensity::num / intensity::den),
        powk(temp::num / temp::den),
        powa(angle::num / angle::den)
        {
        }

        // Some operator overloading
        constexpr quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& 
        operator+=(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A> rhs)
        {
            value += rhs.value;
            return *this;
        }

        constexpr quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& 
        operator-=(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A> rhs)
        {
            value -= rhs.value;
            return *this;
        }

        constexpr quantity<P, m, l, t, q, temp, intensity, angle,  M, L, T, Q, K, I, A>& 
        operator*=(const double rhs)
        {
            value *= rhs;
            return *this;
        }

        constexpr quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& 
        operator/=(const double rhs)
        {
            value /= rhs;
            return *this;
        }

        template<
        Mass_t M2         ,               // Mass unit type
        Length_t L2       ,       // Length unit type
        Time_t T2         ,             // Time unit type
        Charge_t Q2       ,      // Charge unit type
        Temperature_t K2  ,      // Temperature unit type
        Irradiance_t I2   ,     // Luminous Intensity unit type
        Angle_t A2        >            // Angle unit type
        quantity<P, m, l, t, q, temp, intensity, angle, M2, L2, T2, Q2, K2, I2, A2>
        to(const quantity<P, m, l, t, q, temp, intensity, angle, M2, L2, T2, Q2, K2, I2, A2> uTransform) const
        {
            using newQuant = quantity<P, m, l, t, q, temp, intensity, angle, M2, L2, T2, Q2, K2, I2, A2>;
            P newvalue = value;
            // // Same mass type, do nothing
            if ((mType == uTransform.mType) && 
                (lType == uTransform.lType) && 
                (tType  == uTransform.tType) &&
                (qType == uTransform.qType) && 
                (tempType == uTransform.tempType) &&
                (radType == uTransform.radType) && 
                (aType == uTransform.aType)
            )
            {
                return quantity<P, m, l, t, q, temp, intensity, angle, M2, L2, T2, Q2, K2, I2, A2>(value / uTransform.value);
            } 
            
            // if constexpr(std::is_integral_v<m>)
            {
                //=========== Mass types conversions
                if constexpr(m::num != 0)
                {
                    if (mType == Mass_t::Gram)
                    {
                        if (uTransform.mType == Mass_t::Kilogram)
                        {
                            const auto cf = Kilogram(Gram(1.0)).value;
                            newvalue *= std::pow(cf, powm);
                        } else if (uTransform.mType == Mass_t::SolarMass)  {
                            const auto cf = SolarMass(Gram(1.0)).value;
                            newvalue *= std::pow(cf, powm);
                        }
                    } else if (mType == Mass_t::Kilogram){
                        if (uTransform.mType == Mass_t::Gram)
                        {
                            const auto cf = Gram(Kilogram(1.0)).value;
                            newvalue *= std::pow(cf, powm);
                        } else if (uTransform.mType == Mass_t::SolarMass) {
                            const auto cf = SolarMass(Kilogram(1.0)).value;
                            newvalue *= std::pow(cf, powm);
                        }
                    } else {
                        if (uTransform.mType == Mass_t::Gram)
                        {
                            const auto cf = Gram(SolarMass(1.0)).value;
                            newvalue *= std::pow(cf, powm);
                        } else if (uTransform.mType == Mass_t::Kilogram) {
                            const auto cf = Kilogram(SolarMass(1.0)).value;
                            newvalue *= std::pow(cf, powm);
                        }
                    }
                }
                if constexpr(l::num != 0)
                {
                    //==========Length type conversion
                    if (lType == Length_t::Centimeter)
                    {
                        if (uTransform.lType == Length_t::Kilometer)
                        {
                            const auto cf = Kilometer(Centimeter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Meter){
                            const auto cf = Meter(Centimeter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Lightyear) {
                            const auto cf = Lightyear(Centimeter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Parsec)  {
                            const auto cf = Parsec(Centimeter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        }
                    } else if (lType == Length_t::Meter){
                        if (uTransform.lType == Length_t::Centimeter)
                        {
                            const auto cf = Centimeter(Meter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if(uTransform.lType == Length_t::Kilometer) {
                            const auto cf = Kilometer(Meter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Lightyear) {
                            const auto cf = Lightyear(Meter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Parsec)  {
                            const auto cf = Parsec(Meter(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        }
                    } else if (lType == Length_t::Kilometer){
                        if (uTransform.lType == Length_t::Centimeter)
                        {
                            const auto cf = Centimeter(Kilometer(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Meter)  {
                            const auto cf = Meter(Kilometer(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Lightyear) {
                            const auto cf = Lightyear(Kilometer(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Parsec)  {
                            const auto cf = Parsec(Kilometer(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        }
                    } else if (lType == Length_t::Lightyear) {
                        if (uTransform.lType == Length_t::Centimeter)
                        {
                            const auto cf = Centimeter(Lightyear(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if(uTransform.lType == Length_t::Meter) {
                            const auto cf = Meter(Lightyear(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Kilometer) {
                            const auto cf = Kilometer(Lightyear(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Parsec)  {
                            const auto cf = Parsec(Lightyear(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        }
                    } else {
                        if (uTransform.lType == Length_t::Centimeter)
                        {
                            const auto cf = Centimeter(Parsec(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if(uTransform.lType == Length_t::Meter) {
                            const auto cf = Meter(Parsec(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Kilometer) {
                            const auto cf = Kilometer(Parsec(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        } else if (uTransform.lType == Length_t::Lightyear)  {
                            const auto cf = Lightyear(Parsec(1.0)).value;
                            newvalue *= std::pow(cf, powl);
                        }
                    }
                }
                if constexpr(t::num != 0)
                {
                    //==========Time type conversion
                    if (tType == Time_t::Second)
                    {
                        if (uTransform.tType == Time_t::Hour)
                        {
                            const auto cf = Hour(Second(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Day) {
                            const auto cf = Day(Second(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Year) {
                            const auto cf = Year(Second(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        }
                    } else if (tType == Time_t::Hour){
                        if (uTransform.tType == Time_t::Second)
                        {
                            const auto cf = Second(Hour(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Day) {
                            const auto cf = Day(Hour(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Year) {
                            const auto cf = Year(Hour(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        }
                    } else if (tType == Time_t::Day){
                        if (uTransform.tType == Time_t::Hour)
                        {
                            const auto cf = Hour(Day(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Second) {
                            const auto cf = Second(Day(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Year) {
                            const auto cf = Year(Day(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        }
                    } else {
                        if (uTransform.tType == Time_t::Hour)
                        {
                            const auto cf = Hour(Year(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Day) {
                            const auto cf = Day(Year(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        } else if (uTransform.tType == Time_t::Second)  {
                            const auto cf = Second(Year(1.0)).value;
                            newvalue *= std::pow(cf, powt);
                        }
                    } 
                }

                if constexpr(temp::num != 0)
                {
                    //==========Temperature type conversion
                    if (tempType == Temperature_t::Kelvin)
                    {
                        if (uTransform.tempType == Temperature_t::Celcius)
                        {
                            const auto cf = Celcius(Kelvin(1.0)).value;
                            newvalue *= std::pow(cf, powk);
                        } else if (uTransform.tempType == Temperature_t::Fahrenheit) {
                            const auto cf = Fahrenheit(Kelvin(1.0)).value;
                            newvalue *= std::pow(cf, powk);
                        } 
                    } else if (tempType == Temperature_t::Celcius){
                        if (uTransform.tempType == Temperature_t::Kelvin)
                        {
                            const auto cf = Kelvin(Celcius(1.0)).value;
                            newvalue *= std::pow(cf, powk);
                        } else if (uTransform.tempType == Temperature_t::Fahrenheit)  {
                            const auto cf = Fahrenheit(Celcius(1.0)).value;
                            newvalue *= std::pow(cf, powk);
                        } 
                    } else {
                        if (uTransform.tempType == Temperature_t::Celcius)
                        {
                            const auto cf = Celcius(Fahrenheit(1.0)).value;
                            newvalue *= std::pow(cf, powk);
                        } else if (uTransform.tempType == Temperature_t::Kelvin)  {
                            const auto cf = Kelvin(Fahrenheit(1.0)).value;
                            newvalue *= std::pow(cf, powk);
                        } 
                    } 
                }

                if constexpr(q::num != 0)
                {
                    //==========Charge type conversion
                    if (qType == Charge_t::StatCoulomb)
                    {
                        const auto cf = Coulomb(StatCoulomb(1.0)).value;
                        newvalue *= std::pow(cf, powq);
                    } else if (uTransform.qType = Charge_t::Coulomb)  {
                        const auto cf = StatCoulomb(Coulomb(1.0)).value;
                        newvalue *= std::pow(cf, powq);
                    }
                }

                if constexpr(angle::num != 0)
                {
                    //==========Charge type conversion
                    if (aType == Angle_t::Radian)
                    {
                        const auto cf = Degree(Radian(1.0)).value;
                        newvalue *= std::pow(cf, powa);
                    } else if (uTransform.aType = Angle_t::Degree)  {
                        const auto cf = Radian(Degree(1.0)).value;
                        newvalue *= std::pow(cf, powa);
                    }
                }

                return newQuant(newvalue / uTransform.value);
            }
            }

        
        P value;

        // Unit types
        Mass_t mType;
        Length_t lType;
        Angle_t aType;
        Temperature_t tempType;
        Time_t tType;
        Irradiance_t radType;
        Charge_t qType;

        // power values as floats
        double powm, powl, powt, powq, powk, powi, powa;
    };

    //============================================================================================
    template<typename T>
    constexpr std::string rat2str()
    {
        const auto num = std::to_string(T::num);
        if constexpr(T::den != 1)
        {
            const auto den = std::to_string(T::den);
            return num + "/" + den;
        }
        return num;
    }

    //=============================================================================================
    template<
    typename P,           // Precision
    typename m,             // power of mass
    typename l,             // power of length
    typename t,             // power of time
    typename q,             // power of charge
    typename temp,          // power of temperature
    typename intensity,     // power of irradiance
    typename angle,         // power of angle
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    std::ostream& operator<<(std::ostream& out, const quantity<P,m,l,t,q,temp,intensity,angle, M, L, T, Q, K, I, A>& ell)
    {
        std::string outstr;
        std::string padding = "";
        int terms = 0;
        if constexpr(m::num != 0)
        {
            const std::string unit_str = mass_dict[ell.mType];
            if (ell.powm != 1)
            {
                const auto power = rat2str<m>();
                outstr += unit_str + "(" + power + ")";
            }   
            else
            {
                 outstr += unit_str;
            }
            terms++;
        }
        if constexpr(l::num != 0)
        {
            const std::string unit_str = length_dict[ell.lType];
            if (terms != 0)
            {
                padding = " ";
            }
            if (ell.powl!= 1)
            {
                const auto power = rat2str<l>();
                outstr += padding + unit_str + "(" + power + ")";
            } else {
                 outstr += padding + unit_str;
            }
            terms++;
        }
        if constexpr(t::num != 0)
        {
            const std::string unit_str = time_dict[ell.tType];
            if (terms != 0)
            {
                padding = " ";
            }

            if (ell.powt != 1)
            {
                const auto power = rat2str<t>();
                outstr += padding + unit_str + "(" + power + ")";
            } else {
                 outstr += padding + unit_str;
            }

            terms++;
        }
        if constexpr(q::num != 0)
        {
            const std::string unit_str = charge_dict[ell.qType];
            if (terms != 0)
            {
                padding = " ";
            }

            if (ell.powq != 1)
            {
                const auto power = rat2str<q>();
                outstr += padding + unit_str + "(" + power + ")";
            } else {
                 outstr += padding + unit_str;
            }
            terms++;
        }
        if constexpr(temp::num != 0)
        {
            const auto  unit_str = temp_dict[ell.tempType];
            if (terms != 0)
            {
                padding = " ";
            }
            if (ell.powk != 1)
            {
                const auto power = rat2str<temp>();
                outstr += padding + unit_str + "(" + power + ")";
            } else {
                 outstr += padding + unit_str;
            }

            terms++;
        }
        if constexpr(intensity::num != 0)
        {
            const auto  unit_str = rad_dict[ell.radType];
            if (terms != 0)
            {
                padding = " ";
            }
            if (ell.powi != 1)
            {
                const auto power = rat2str<intensity>();
                outstr += padding + unit_str + "(" + power + ")";
            } else {
                 outstr += padding + unit_str;
            }
            terms++;
        }
        if constexpr(angle::num != 0)
        {
            const auto  unit_str = angle_dict[ell.aType];
            if (terms != 0)
            {
                padding = " ";
            }
            if (ell.powa != 1)
            {
                const auto power = rat2str<angle>();
                outstr += padding + unit_str + "(" + power + ")";
            } else {
                 outstr += padding + unit_str;
            }
            terms++;
        }
        out << ell.value << "[" << outstr << "]";
        return out;
    }

    // "Non-assignment operators are best implemented as non-members"
    template<
    typename P,
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M         ,      // Mass unit type
    Length_t L       ,      // Length unit type
    Time_t T         ,      // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,      // Luminous Intensity unit type
    Angle_t A>             // Angle unit type
    constexpr auto
    operator<(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& lhs,
              const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& rhs)
    {
        return lhs.value < rhs.value;
    }
    
    template<
    typename P,
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M         ,    // Mass unit type
    Length_t L       ,    // Length unit type
    Time_t T         ,    // Time unit type
    Charge_t Q       ,    // Charge unit type
    Temperature_t K  ,    // Temperature unit type
    Irradiance_t I   ,    // Luminous Intensity unit type
    Angle_t A>            // Angle unit type
    constexpr auto
    operator>(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& lhs,
              const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& rhs)
    {
        return lhs.value > rhs.value;
    }

    template<
    typename P,
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    constexpr auto operator+(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& lhs,
    const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& rhs)
    {
        quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A> result(lhs);
        return result += rhs;
    }

    template<
    typename P,
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    constexpr auto operator-(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& lhs,
              const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& rhs)
    {
        quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A> result(lhs);
        return result -= rhs;
    }

    template<
    typename P, 
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M,       // Mass unit type
    Length_t L,       // Length unit type
    Time_t T,       // Time unit type
    Charge_t Q,       // Charge unit type
    Temperature_t K,       // Temperature unit type
    Irradiance_t I,       // Luminous Intensity unit type
    Angle_t A        >       // Angle unit type
    constexpr auto operator*(double lhs,
                             const quantity<P, m, l, t,q, temp, intensity, angle, M, L, T, Q, K, I, A>& rhs)
    {
        quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A> result(rhs);
        return result *= lhs;
    }

    template<
    typename P, 
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M ,                  // Mass unit type
    Length_t L ,        // Length unit type
    Time_t T ,                // Time unit type
    Charge_t Q ,       // Charge unit type
    Temperature_t K ,  // Temperature unit type
    Irradiance_t I ,  // Luminous Intensity unit type
    Angle_t A >             // Angle unit type
    constexpr auto operator*(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& lhs, double rhs)
    {
        quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A> result(lhs);
        return result *= rhs;
    }

    template<
    typename P, 
    typename m, 
    typename l, 
    typename t, 
    typename q, 
    typename temp, 
    typename intensity, 
    typename angle,
    Mass_t M ,                  // Mass unit type
    Length_t L ,        // Length unit type
    Time_t T ,                // Time unit type
    Charge_t Q ,       // Charge unit type
    Temperature_t K ,  // Temperature unit type
    Irradiance_t I ,  // Luminous Intensity unit type
    Angle_t A >             // Angle unit type
    constexpr auto operator/(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& lhs, double rhs)
    {
        return quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>(lhs.value / rhs);
    }

    template<
    typename P, 
    typename m1, typename l1, typename t1, typename q1, typename temp1, typename intensity1, typename angle1,
    typename m2, typename l2, typename t2, typename q2, typename temp2, typename intensity2, typename angle2,
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    constexpr auto operator*(const quantity<P, m1, l1, t1, q1, temp1, intensity1, angle1,M, L, T, Q, K, I, A>& lhs,
              const quantity<P, m2, l2, t2, q2, temp2, intensity2, angle2, M, L, T, Q, K, I, A>& rhs)
    {
        typedef quantity<P, 
        std::ratio_add<m1,m2>, 
        std::ratio_add<l1,l2>, 
        std::ratio_add<t1,t2>, 
        std::ratio_add<q1,q2>, 
        std::ratio_add<temp1,temp2>,
        std::ratio_add<intensity1,intensity2>, 
        std::ratio_add<angle1,angle2>, M, L, T, Q, K, I, A> ResultType;
        
        return ResultType(lhs.value * rhs.value);
    }

    template<
    typename P, 
    typename m1, typename l1, typename t1, typename q1, typename temp1, typename intensity1, typename angle1,
    typename m2, typename l2, typename t2, typename q2, typename temp2, typename intensity2, typename angle2,
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    constexpr auto operator-(const quantity<P, m1, l1, t1, q1, temp1, intensity1, angle1,M, L, T, Q, K, I, A>& lhs,
              const quantity<P, m2, l2, t2, q2, temp2, intensity2, angle2, M, L, T, Q, K, I, A>& rhs)
    {
        typedef quantity<P, 
        std::ratio_add<m1,m2>, 
        std::ratio_add<l1,l2>, 
        std::ratio_add<t1,t2>, 
        std::ratio_add<q1,q2>, 
        std::ratio_add<temp1,temp2>,
        std::ratio_add<intensity1,intensity2>, 
        std::ratio_add<angle1,angle2>, M, L, T, Q, K, I, A> ResultType;
        
        return ResultType(lhs.value * rhs.value);
    }
    
    template<
    typename P, 
    typename m1, typename l1, typename t1, typename q1, typename temp1, typename intensity1, typename angle1,
    typename m2, typename l2, typename t2, typename q2, typename temp2, typename intensity2, typename angle2,
    Mass_t M         ,               // Mass unit type
    Length_t L       ,       // Length unit type
    Time_t T         ,             // Time unit type
    Charge_t Q       ,      // Charge unit type
    Temperature_t K  ,      // Temperature unit type
    Irradiance_t I   ,     // Luminous Intensity unit type
    Angle_t A        >            // Angle unit type
    constexpr auto operator/(const quantity<P, m1, l1, t1, q1, temp1, intensity1, angle1, M, L, T, Q, K, I, A>& lhs,
              const quantity<P, m2, l2, t2, q2, temp2, intensity2, angle2, M, L, T, Q, K, I, A>& rhs)
    {
        typedef quantity<P, 
        std::ratio_subtract<m1,m2>,
        std::ratio_subtract<l1,l2>,
        std::ratio_subtract<t1,t2>, 
        std::ratio_subtract<q1,q2>,
        std::ratio_subtract<temp1,temp2>, 
        std::ratio_subtract<intensity1,intensity2>, 
        std::ratio_subtract<angle1,angle2>, 
        M, L, T, Q, K, I, A> ResultType;
        return ResultType(lhs.value / rhs.value);
    }

    template<
    typename P, 
    typename m1, typename l1, typename t1, typename q1, typename temp1, typename intensity1, typename angle1,
    typename m2, typename l2, typename t2, typename q2, typename temp2, typename intensity2, typename angle2,
    Mass_t M1, Mass_t M2, Length_t L1, Length_t L2,       
    Time_t T1, Time_t T2, Charge_t Q1,Charge_t Q2,       
    Temperature_t K1, Temperature_t K2, Irradiance_t I1, Irradiance_t I2,
    Angle_t A1, Angle_t A2>              
    constexpr auto operator/(const quantity<P, m1, l1, t1, q1, temp1, intensity1, angle1, M1, L1, T1, Q1, K1, I1, A1>& lhs,
                         const quantity<P, m2, l2, t2, q2, temp2, intensity2, angle2, M2, L2, T2, Q2, K2, I2, A2>& rhs)
    {
        const Mass_t new_mType           = (M1 > M2) ? M1 : M2;
        const Length_t new_lType         = (L1 > L2) ? L1 : L2;
        const Time_t new_tType           = (T1 > T2) ? T1 : T2;
        const Temperature_t new_tempType = (K1 > K2) ? K1 : K2;
        const Irradiance_t new_radType   = (I1 > I2) ? I1 : I2;
        const Angle_t new_aType          = (A1 > A2) ? A1 : A2;
        const Charge_t new_qType         = (Q1 > Q2) ? Q1 : Q2;

        return quantity<P, 
                std::ratio_subtract<m1,m2>,
                std::ratio_subtract<l1,l2>,
                std::ratio_subtract<t1,t2>, 
                std::ratio_subtract<q1,q2>,
                std::ratio_subtract<temp1,temp2>, 
                std::ratio_subtract<intensity1,intensity2>, 
                std::ratio_subtract<angle1,angle2>,
                new_mType, new_lType, new_tType, new_qType, new_tempType, new_radType, new_aType>(lhs.value / rhs.value);
    }

    // // Special case for dimensionless type 
    template<typename P>
    struct quantity<P, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>> {
        constexpr quantity(P initVal = 0): value(initVal) {}        // allow implicit conversion
        constexpr operator P() const { return value; }    // to/from values of type P
        constexpr quantity& operator=(P newVal)                     // allow assignments from
        { value = newVal; return *this; }                 // values of type T
        P value;
    };

    namespace math
    {
        template<
        typename power,
        typename P, 
        typename m, 
        typename l, 
        typename t, 
        typename q, 
        typename temp, 
        typename intensity, 
        typename angle,
        Mass_t M         ,               // Mass unit type
        Length_t L       ,       // Length unit type
        Time_t T         ,             // Time unit type
        Charge_t Q       ,      // Charge unit type
        Temperature_t K  ,      // Temperature unit type
        Irradiance_t I   ,     // Luminous Intensity unit type
        Angle_t A        >            // Angle unit type
        constexpr auto pow(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& quant)
        {
            const P vpower = P(power::num) / power::den;
            return  quantity<P, std::ratio_multiply<m, power>, 
                                std::ratio_multiply<l, power>, 
                                std::ratio_multiply<t, power>,
                                std::ratio_multiply<q, power>,
                                std::ratio_multiply<temp, power>,
                                std::ratio_multiply<intensity, power>,
                                std::ratio_multiply<angle, power>,
                                M, L, T, Q, K, I, A>(std::pow(quant.value, vpower));
        }

        template<
        typename P, 
        typename m, 
        typename l, 
        typename t, 
        typename q, 
        typename temp, 
        typename intensity, 
        typename angle,
        Mass_t M         ,               // Mass unit type
        Length_t L       ,       // Length unit type
        Time_t T         ,             // Time unit type
        Charge_t Q       ,      // Charge unit type
        Temperature_t K  ,      // Temperature unit type
        Irradiance_t I   ,     // Luminous Intensity unit type
        Angle_t A        >            // Angle unit type
        constexpr auto sqrt(const quantity<P, m, l, t, q, temp, intensity, angle, M, L, T, Q, K, I, A>& val)
        {
            return  quantity<P, std::ratio_divide<m, std::ratio<2>>, 
                                std::ratio_divide<l, std::ratio<2>>, 
                                std::ratio_divide<t, std::ratio<2>>,
                                std::ratio_divide<q, std::ratio<2>>,
                                std::ratio_divide<temp, std::ratio<2>>,
                                std::ratio_divide<intensity, std::ratio<2>>,
                                std::ratio_divide<angle, std::ratio<2>>,
                                M, L, T, Q, K, I, A>( std::sqrt(val.value) );
        }
    } // namespace math

     // define the physical units
    // Base units
    using mass          = quantity<double, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using length        = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using mytime        = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using temp          = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>>;
    using intensity     = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>>;
    using angle         = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>>;

    // // Derived units
    using velocity      = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<-1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using acceleration  = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using energy        = quantity<double, std::ratio<1>, std::ratio<2>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using frequency     = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<-1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using force         = quantity<double, std::ratio<1>, std::ratio<1>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using spectral_flux = quantity<double, std::ratio<1>, std::ratio<0>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using total_flux    = quantity<double, std::ratio<1>, std::ratio<0>, std::ratio<-3>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using power         = quantity<double, std::ratio<1>, std::ratio<2>,std::ratio<-3>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using volume        = quantity<double, std::ratio<0>, std::ratio<3>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using area          = quantity<double, std::ratio<0>, std::ratio<2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using mag_field     = quantity<double, std::ratio<1,2>, std::ratio<-1,2>, std::ratio<-1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using charge        = quantity<double, std::ratio<1,2>, std::ratio<3,2>, std::ratio<-1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using edens         = quantity<double, std::ratio<1>, std::ratio<-1>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using mdens         = quantity<double, std::ratio<1>, std::ratio<-3>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using ndens         = quantity<double, std::ratio<0>, std::ratio<-3>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using spec_power    = quantity<double, std::ratio<1>, std::ratio<2>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;
    using emissivity    = quantity<double, std::ratio<1>, std::ratio<-1>, std::ratio<-2>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>>;

    // define some commaon cgs units
    constexpr mass          gram(1);                   // gram
    constexpr length        cm(1);                     // centimeter
    constexpr mytime        s(1);                      // second
    constexpr charge        statC(1);                  // statColoumb
    constexpr temp          kelvin(1);                 // Kelvin
    constexpr angle         rad(1);                    // radians 
    constexpr energy        erg(1);                    // erg
    constexpr frequency     hz(1);                     // Hertz
    constexpr force         dyne(1);                   // dyne
    constexpr spectral_flux jy(1e-23);                 // Jansky
    constexpr spectral_flux mjy(1e-26);               // millJansky
    constexpr volume        cm3(1);                    // centimeters cubed
    constexpr area          cm2(1);                    // centimeters squared
    constexpr mag_field     gauss(1);                  // Gauss units
    constexpr edens         erg_per_cm3(1);            // erg per centimer cubed
    constexpr mdens         g_per_cm3(1);              // gram per centimer cubed
    constexpr power         erg_per_s(1);              // erg per second
    constexpr ndens         n_per_cm3(1);              // number of particles per centimer cubed
    constexpr spec_power    power_per_hz(1);           // erg per second squared
    constexpr emissivity    power_per_hz_per_cm3(1);   // erg per second squared per centimer cubed

    // // define derived conversion types
    constexpr auto kg    = quantity<double, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Kilogram>(1);
    constexpr auto mSun  = quantity<double, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::SolarMass>(1);

    // distance
    constexpr auto km = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Kilometer>(1);
    constexpr auto m  = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Meter>(1);
    constexpr auto ly = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Lightyear>(1);
    constexpr auto pc = quantity<double, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Parsec>(1);

    // time
    constexpr auto day  = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Centimeter, Time_t::Day>(1);
    constexpr auto hour = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Centimeter, Time_t::Hour>(1);
    constexpr auto year = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Centimeter, Time_t::Year>(1);

    // charge
    constexpr auto coulomb  = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Centimeter, Time_t::Second, Charge_t::Coulomb>(1);

    // temperature
    constexpr auto celcius    = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Centimeter, Time_t::Second, Charge_t::StatCoulomb, Temperature_t::Celcius>(1);
    constexpr auto fahrenheit = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<1>, std::ratio<0>, std::ratio<0>, Mass_t::Gram, Length_t::Centimeter, Time_t::Second, Charge_t::StatCoulomb, Temperature_t::Fahrenheit>(1);

    // angle
    constexpr auto deg = quantity<double, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>, std::ratio<0>,std::ratio<1>, Mass_t::Gram, Length_t::Centimeter, Time_t::Second, Charge_t::StatCoulomb, Temperature_t::Kelvin, Irradiance_t::ErgCM2P2, Angle_t::Degree>(1);

} // namespace units