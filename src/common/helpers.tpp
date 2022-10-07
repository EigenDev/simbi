/**
* Helper Function Template Tool
*/

namespace simbi
{
    namespace helpers
    {
        template <typename T, size_t N>
        constexpr size_t array_size(T (&)[N]) {
            return N;
        }

        template<typename ... Args>
        std::string string_format( const std::string& format, Args ... args )
        {
            size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
            if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
            std::unique_ptr<char[]> buf( new char[ size ] ); 
            snprintf( buf.get(), size, format.c_str(), args ... );
            return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
        }

        //Handle 2D primitive arrays whether SR or Newtonian
        template<typename T, typename N>
        typename std::enable_if<is_3D_primitive<N>::value>::type
        writeToProd(T *from, PrimData *to){
            to->rho  = from->rho;
            to->v1   = from->v1;
            to->v2   = from->v2;
            to->v3   = from->v3;
            to->p    = from->p;
            to->chi  = from->chi;
        }

        //Handle 2D primitive arrays whether SR or Newtonian
        template<typename T, typename N>
        typename std::enable_if<is_2D_primitive<N>::value>::type
        writeToProd(T *from, PrimData *to){
            to->rho  = from->rho;
            to->v1   = from->v1;
            to->v2   = from->v2;
            to->p    = from->p;
            to->chi  = from->chi;
        }

        template<typename T, typename N>
        typename std::enable_if<is_1D_primitive<N>::value>::type
        writeToProd(T *from, PrimData *to){
            to->rho  = from->rho;
            to->v    = from->v;
            to->p    = from->p;
        }

        template<typename T , typename N>
        typename std::enable_if<is_3D_primitive<N>::value, T>::type
        vec2struct(const std::vector<N> &p){
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.v3.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++)
            {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.v3.push_back(p[i].v3);
                sprims.p.push_back(p[i].p);
                sprims.chi.push_back(p[i].chi);
            }
            
            return sprims;
        }

        template<typename T , typename N>
        typename std::enable_if<is_2D_primitive<N>::value, T>::type
        vec2struct(const std::vector<N> &p){
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++)
            {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.p.push_back(p[i].p);
                sprims.chi.push_back(p[i].chi);
            }
            
            return sprims;
        }

        template<typename T , typename N>
        typename std::enable_if<is_1D_primitive<N>::value, T>::type
        vec2struct(const std::vector<N> &p){
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v.reserve(nzones);
            sprims.p.reserve(nzones);
            for (size_t i = 0; i < nzones; i++)
            {
                sprims.rho.push_back(p[i].rho);
                sprims.v.push_back(p[i].v);
                sprims.p.push_back(p[i].p);
            }
            
            return sprims;
        }

        template<typename T, typename U, typename V, typename W, int X>
        void write_to_file(
            T *sim_state_host, 
            T *sim_state_dev, 
            W  &dual_memory_layer,
            DataWriteMembers &setup,
            const std::string data_directory,
            const real t, 
            const real t_interval, 
            const real chkpt_interval, 
            const luint chkpt_zone_label)
        {
            if constexpr(BuildPlatform == Platform::GPU) {
                dual_memory_layer.copyDevToHost(sim_state_dev, *sim_state_host);
            }
            setup.x1max = sim_state_host->x1max;
            setup.x1min = sim_state_host->x1min;

            PrimData prods;
            static auto step                = sim_state_host->init_chkpt_idx;
            static autp tbefore             = sim_state_host->tstart;
            static std::string tchunk       = "000000";
            static lint tchunk_order_of_mag = 2;
            const auto time_order_of_mag    = std::floor(std::log10(t));
            if (time_order_of_mag > tchunk_order_of_mag) {
                tchunk.insert(0, "0");
                tchunk_order_of_mag += 1;
            }

            // Transform vector of primitive structs to struct of primitive vectors
            auto transfer_prims = vec2struct<U, V>(sim_state_host->prims);
            writeToProd<U, V>(&transfer_prims, &prods);
            std::string tnow;
            if (sim_state_host->dlogt != 0)
            {
                const auto time_order_of_mag = std::floor(std::log10(step));
                if (time_order_of_mag > tchunk_order_of_mag) {
                    tchunk.insert(0, "0");
                    tchunk_order_of_mag += 1;
                }
                tnow = create_step_str(step, tchunk);
            } else {
                tnow = create_step_str(t_interval, tchunk);
            }
            if (t_interval == INFINITY) {
                tnow = "interrupted";
            }
            const auto filename = string_format("%d.chkpt." + tnow + ".h5", chkpt_zone_label);

            setup.t             = t;
            setup.dt            = t - tbefore;
            setup.chkpt_idx     = step;
            tbefore             = t;
            step++;
            write_hdf5(data_directory, filename, prods, setup, X, sim_state_host->total_zones);
        }
        
    } // namespace helpers
    
} // namespace simbi