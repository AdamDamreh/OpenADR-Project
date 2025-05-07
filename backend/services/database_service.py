import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone, timedelta # Ensure timezone and timedelta are imported
from contextlib import asynccontextmanager

# Import the Pydantic model to use its structure
try:
    from backend.main import PowerReading
except ImportError:
    # Fallback or alternative definition if direct import causes issues
    from pydantic import BaseModel, Field

    class PowerReading(BaseModel): # Simplified fallback
        timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        power_watts: float
        device_id: str | None = None


logger = logging.getLogger(__name__)

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./simulation_data.db") # Default to SQLite async
# Ensure the URL uses the async driver format
if DATABASE_URL.startswith("sqlite:///"):
    DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
elif DATABASE_URL.startswith("postgresql://"): # Handle potential future switch
     DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


logger.info(f"Using database URL: {DATABASE_URL}")

# --- SQLAlchemy Setup ---
engine = create_async_engine(DATABASE_URL, echo=False) # Set echo=True for SQL query logging
# Define naming convention for constraints to avoid issues with SQLite ALTER TABLE limitations
metadata_obj = MetaData(naming_convention={
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
})
Base = declarative_base(metadata=metadata_obj)

# Async Session Maker
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False, # Important for async context
    autocommit=False,
    autoflush=False,
)

# --- Database Models ---

# Original model (if needed elsewhere, otherwise could be removed if unused)
class PowerReadingDB(Base):
    __tablename__ = "power_readings" # Original table

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    power_watts = Column(Float, nullable=False)
    device_id = Column(String, nullable=True, index=True)

# New model for live incoming data
class LivePowerReadingDB(Base):
    __tablename__ = "live_power_readings" # New table for live data

    id = Column(Integer, primary_key=True, index=True)
    # Ensure DateTime stores timezone info if the input has it
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    power_watts = Column(Float, nullable=False)
    device_id = Column(String, nullable=True, index=True)

# Model for OpenADR events
class OpenADREventDB(Base):
    __tablename__ = "openadr_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, nullable=False, index=True, unique=True)
    signal_type = Column(String, nullable=False)
    signal_level = Column(Float, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    target_ven_id = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")  # active, completed, cancelled
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))


# --- Database Initialization ---
async def init_db():
    """Initializes the database and creates tables if they don't exist."""
    async with engine.begin() as conn:
        try:
            logger.info("Initializing database and creating tables...")
            # await conn.run_sync(Base.metadata.drop_all) # Uncomment to drop tables on startup
            # This will create BOTH tables ('power_readings' and 'live_power_readings') if they don't exist
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables checked/created successfully.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            raise # Re-raise the exception to potentially halt startup if critical

# --- Database Session Dependency ---
@asynccontextmanager
async def get_db_session() -> AsyncSession:
    """Provides a database session for dependency injection."""
    session = AsyncSessionFactory()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database error occurred: {e}", exc_info=True)
        await session.rollback()
        raise # Re-raise after rollback
    except Exception as e:
         logger.error(f"An unexpected error occurred in DB session: {e}", exc_info=True)
         await session.rollback()
         raise # Re-raise after rollback
    finally:
        await session.close()

# --- Data Saving Function ---
async def save_power_reading(reading_data: PowerReading):
    """Saves a PowerReading object to the 'live_power_readings' table."""
    logger.debug(f"Attempting to save live power reading: {reading_data.model_dump()}")
    async with get_db_session() as session:
        try:
            # Ensure timestamp is timezone-aware UTC before saving
            ts = reading_data.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)

            # Use the NEW model for live data
            db_reading = LivePowerReadingDB(
                timestamp=ts,
                power_watts=reading_data.power_watts,
                device_id=reading_data.device_id
            )
            session.add(db_reading)
            # await session.commit() # Commit is handled by the context manager
            await session.flush() # Flush to get potential errors before commit
            await session.refresh(db_reading) # Refresh to get the ID assigned by the DB
            logger.info(f"Successfully saved live power reading with ID: {db_reading.id} to table '{LivePowerReadingDB.__tablename__}'")
            return db_reading # Return the saved object with its ID
        except Exception as e:
            # Error logging is handled by get_db_session context manager
            return None # Indicate failure

# --- Data Fetching Functions ---
async def get_latest_live_power_reading() -> LivePowerReadingDB | None:
    """Fetches the most recent power reading from the 'live_power_readings' table."""
    logger.debug(f"Attempting to fetch the latest reading from '{LivePowerReadingDB.__tablename__}'...")
    async with get_db_session() as session:
        try:
            from sqlalchemy.future import select
            stmt = select(LivePowerReadingDB).order_by(LivePowerReadingDB.timestamp.desc()).limit(1)
            result = await session.execute(stmt)
            latest_reading = result.scalar_one_or_none()
            if latest_reading:
                logger.info(f"Successfully fetched latest live power reading with ID: {latest_reading.id}")
            else:
                logger.warning(f"No readings found in '{LivePowerReadingDB.__tablename__}'.")
            return latest_reading
        except Exception as e:
            # Error logging handled by context manager
            return None

async def get_all_live_power_readings() -> list[LivePowerReadingDB]:
    """Fetches all power readings from the 'live_power_readings' table."""
    logger.debug(f"Attempting to fetch all readings from '{LivePowerReadingDB.__tablename__}'...")
    readings = []
    async with get_db_session() as session:
        try:
            from sqlalchemy.future import select
            stmt = select(LivePowerReadingDB).order_by(LivePowerReadingDB.timestamp.asc()) # Fetch in chronological order
            result = await session.execute(stmt)
            readings = result.scalars().all()
            logger.info(f"Successfully fetched {len(readings)} readings from '{LivePowerReadingDB.__tablename__}'.")
        except Exception as e:
            # Error logging handled by context manager
            logger.error(f"Failed to fetch all live readings: {e}", exc_info=True)
            # Return empty list on failure
    return readings

# --- OpenADR Event Functions ---
async def save_openadr_event(event_data):
    """Saves an OpenADR event to the database."""
    logger.debug(f"Attempting to save OpenADR event: {event_data}")
    async with get_db_session() as session:
        try:
            # Calculate end time based on start time and duration
            start_time = event_data["start_time"]
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            
            duration_minutes = event_data["duration_minutes"]
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            # Create the database model instance
            db_event = OpenADREventDB(
                event_id=event_data["event_id"],
                signal_type=event_data["signal_type"],
                signal_level=event_data["signal_level"],
                start_time=start_time,
                end_time=end_time,
                duration_minutes=duration_minutes,
                target_ven_id=event_data.get("target_ven_id"),
                status=event_data.get("status", "active")
            )
            
            session.add(db_event)
            await session.flush()
            await session.refresh(db_event)
            logger.info(f"Successfully saved OpenADR event with ID: {db_event.id}, event_id: {db_event.event_id}")
            return db_event
        except Exception as e:
            logger.error(f"Error saving OpenADR event: {e}", exc_info=True)
            return None

async def get_active_openadr_events():
    """Fetches all active OpenADR events (current and future)."""
    logger.debug("Attempting to fetch active OpenADR events...")
    events = []
    async with get_db_session() as session:
        try:
            from sqlalchemy.future import select
            now = datetime.now(timezone.utc)
            # Get events that are active (end time is in the future)
            stmt = select(OpenADREventDB).where(
                OpenADREventDB.end_time >= now,
                OpenADREventDB.status == "active"
            ).order_by(OpenADREventDB.start_time.asc())
            
            result = await session.execute(stmt)
            events = result.scalars().all()
            logger.info(f"Successfully fetched {len(events)} active OpenADR events.")
        except Exception as e:
            logger.error(f"Failed to fetch active OpenADR events: {e}", exc_info=True)
    return events

async def get_all_openadr_events(limit=50):
    """Fetches all OpenADR events, with optional limit."""
    logger.debug(f"Attempting to fetch all OpenADR events (limit={limit})...")
    events = []
    async with get_db_session() as session:
        try:
            from sqlalchemy.future import select
            stmt = select(OpenADREventDB).order_by(OpenADREventDB.start_time.desc()).limit(limit)
            result = await session.execute(stmt)
            events = result.scalars().all()
            logger.info(f"Successfully fetched {len(events)} OpenADR events.")
        except Exception as e:
            logger.error(f"Failed to fetch OpenADR events: {e}", exc_info=True)
    return events

async def update_openadr_event_status(event_id, new_status):
    """Updates the status of an OpenADR event."""
    logger.debug(f"Attempting to update OpenADR event {event_id} status to {new_status}...")
    async with get_db_session() as session:
        try:
            from sqlalchemy.future import select
            stmt = select(OpenADREventDB).where(OpenADREventDB.event_id == event_id)
            result = await session.execute(stmt)
            event = result.scalar_one_or_none()
            
            if event:
                event.status = new_status
                event.updated_at = datetime.now(timezone.utc)
                await session.flush()
                logger.info(f"Successfully updated OpenADR event {event_id} status to {new_status}")
                return True
            else:
                logger.warning(f"OpenADR event {event_id} not found for status update")
                return False
        except Exception as e:
            logger.error(f"Failed to update OpenADR event status: {e}", exc_info=True)
            return False


if __name__ == '__main__':
    # Example usage (for testing this module directly)
    import asyncio

    async def test_db():
        logging.basicConfig(level=logging.INFO)
        logger.info("Running database service test...")
        await init_db() # This will ensure both tables exist

        # Create a test reading
        test_reading = PowerReading(
            power_watts=555.66,
            device_id="live-test-device-002"
            # Timestamp will be defaulted by Pydantic model
        )

        logger.info("Attempting to save test reading to live table...")
        saved = await save_power_reading(test_reading) # This now saves to live_power_readings

        if saved:
            logger.info(f"Test reading saved successfully to '{LivePowerReadingDB.__tablename__}': ID={saved.id}, Timestamp={saved.timestamp}")
        else:
            logger.error("Failed to save test reading.")

        # Example query from the NEW table
        async with get_db_session() as session:
            from sqlalchemy.future import select
            logger.info(f"Querying last 5 readings from '{LivePowerReadingDB.__tablename__}'...")
            result = await session.execute(
                select(LivePowerReadingDB).order_by(LivePowerReadingDB.timestamp.desc()).limit(5)
            )
            readings = result.scalars().all()
            logger.info(f"Found {len(readings)} readings in '{LivePowerReadingDB.__tablename__}':")
            for r in readings:
                logger.info(f"  - ID: {r.id}, Time: {r.timestamp}, Watts: {r.power_watts}, Device: {r.device_id}")

    asyncio.run(test_db())
