import java.util.concurrent.Semaphore;

public class ReadersWriters {
    private static int value = 0;
    private static final int N = 7;

    private static final Semaphore mutex = new Semaphore(1);

    private static final Semaphore readersDone = new Semaphore(0);

    private static final Semaphore writerDoneForReader1 = new Semaphore(0);
    private static final Semaphore writerDoneForReader2 = new Semaphore(0);

    private static int readCount = 0;

    public static void main(String[] args) throws InterruptedException {
        final int rounds = N; 

        Thread reader1 = new Thread(() -> readerTask("Reader-1", rounds, writerDoneForReader1));
        Thread reader2 = new Thread(() -> readerTask("Reader-2", rounds, writerDoneForReader2));
        Thread writer = new Thread(() -> writerTask(rounds));

        reader1.start();
        reader2.start();
        writer.start();

        reader1.join();
        reader2.join();
        writer.join();
    }

    private static void readerTask(String name, int rounds, Semaphore myWriterDone) {
        for (int i = 0; i < rounds; i++) {
            try {
                if (i > 0) {
                    myWriterDone.acquire();
                }

                System.out.println(name + " reads " + value);

                mutex.acquire();
                readCount++;
                if (readCount == 2) {
                    readCount = 0;      
                    readersDone.release();  
                }
                mutex.release();

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private static void writerTask(int rounds) {
        for (int i = 0; i < rounds; i++) {
            try {
                readersDone.acquire();

                value++;
                System.out.println("Writer writes " + value);

                writerDoneForReader1.release();
                writerDoneForReader2.release();

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}