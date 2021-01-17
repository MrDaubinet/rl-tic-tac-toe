<script>
	// store
  import history from "../stores/History.ts"
  // store information
  import { game } from "../game.js"

  // function prop
  export let updateState

  // Go back in game time
  async function goBack(index) {
    // remove states from history
    history.selectState(index)
    // update game state to the histories latest state
    game.setStates($history[$history.length - 1].state)
    // update game turn to the histories latest turn
    game.setTurn($history[$history.length - 1].turn)
    // update the board state
    updateState()
  }
</script>


{#each $history as state, index}
  {#if index != ($history.length -1)}
    <div class="{index == 0 ? 'pb-4' : 'py-4'}">
      <div on:click={() => goBack(index)}
        class="flex justify-center text-2xl cursor-pointer hover:underline">
        <h1> Go Back, </h1>
        <div class="ml-2">
          {#if index == 0}
            to start
          {:else}
            to move {index }
          {/if}
        </div>
      </div>
    </div>
  {/if}
{/each}
