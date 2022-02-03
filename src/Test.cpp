#include "Test.hpp"

Test::Test()
{

}

bool Test::camera_test()
{
   
    Camera cam(cap, frame);
    return(cam.run_camera());
}

bool Test::build_engine()
{
    Configurations config;
    config.optBatchSize = {2, 4, 8};

    clock_t start, end;
    double time;


    Engine engine(config);

    bool succ = engine.build(MODEL);
    if(!succ)
    {
        throw runtime_error("Could not built TRT engine");
    }
    succ = engine.loadNetwork();
    if(!succ)
    {
         throw runtime_error("Could not load network");
    }
    // start = clock();
    // succ = engine.inference();
    // end = clock();
    // if(!succ)
    // {
    //     throw runtime_error("Could not run inference");
    // }
    // time = ((double)end - double(start))/CLOCKS_PER_SEC;
    // printf("Time of inference process: %f\n", time);
    // printf("End of code\n");

    return succ;
}