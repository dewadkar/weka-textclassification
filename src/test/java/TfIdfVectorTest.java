import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by dewadkar on 8/3/2016.
 */
public class TfIdfVectorTest {

    @Test
    public void termFrequencyTest(){
        TfIdfVector tfIdfVector = new TfIdfVector();
        assertEquals(0,tfIdfVector.termFrequency(),0);
    }
}
